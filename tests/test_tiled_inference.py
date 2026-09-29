from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from schemas.data_base import InputParamsBusiness
from schemas.exceptions import ModelInferenceError
from services.inference import TensorInfo
from vie_plugin_dc_fuse.business_logic import DCFuseDetectorAPI
from vie_plugin_dc_fuse.config import DcFuseConfig
from vie_plugin_dc_fuse.dc_fuse_detect import DCFuseDetector


def test_mode_specific_model_and_confidence_defaults():
    original = DcFuseConfig(_env_file=None, tiled_inference=False)
    tiled = DcFuseConfig(_env_file=None, tiled_inference=True)
    assert original.inference_model_path == original.model_path
    assert original.inference_conf_threshold == original.confThreshold == 0.6
    assert tiled.inference_model_path.endswith('det_yolo_v6_split.onnx')
    assert tiled.inference_conf_threshold == 0.4


def test_tiled_settings_from_environment(monkeypatch):
    monkeypatch.setenv('DC_FUSE_TILED_INFERENCE', 'true')
    monkeypatch.setenv('DC_FUSE_TILED_CONF_THRESHOLD', '0.5')
    monkeypatch.setenv('DC_FUSE_TILE_OVERLAP', '100')
    cfg = DcFuseConfig(_env_file=None)
    assert cfg.tiled_inference and cfg.inference_conf_threshold == 0.5
    assert cfg.tile_overlap == 100


@pytest.mark.parametrize('name,value', [
    ('tiled_inference', 'invalid'), ('tiled_conf_threshold', -0.1),
    ('tiled_conf_threshold', 1.1), ('tile_overlap', -1), ('tile_overlap', 'bad'),
    ('tile_overlap', 2.5), ('tiled_model_path', ''),
])
def test_invalid_tiled_settings(name, value):
    with pytest.raises(ValidationError):
        DcFuseConfig(_env_file=None, **{name: value})


def test_initialization_selects_split_runner_and_tiled_detector(monkeypatch):
    monkeypatch.setenv('DC_FUSE_TILED_INFERENCE', 'true')
    runner = MagicMock()
    with (
        patch('vie_plugin_dc_fuse.business_logic.create_inference_runner', return_value=runner) as factory,
        patch('vie_plugin_dc_fuse.business_logic.DCFuseDetector') as detector,
    ):
        DCFuseDetectorAPI(MagicMock())
    assert factory.call_args.args[0].onnx_path.endswith('det_yolo_v6_split.onnx')
    detector.assert_called_once_with(
        runner, 0.4, tiled_inference=True, tile_overlap=200,
        class_conf_thresholds=DcFuseConfig(_env_file=None).tiled_class_conf_thresholds.model_dump(),
        copper_max_aspect_ratio=1.0,
    )


def test_split_load_failure_never_retries_original_model(monkeypatch):
    monkeypatch.setenv('DC_FUSE_TILED_INFERENCE', 'true')
    with patch('vie_plugin_dc_fuse.business_logic.create_inference_runner',
               side_effect=RuntimeError('missing split')) as factory:
        with pytest.raises(ModelInferenceError):
            DCFuseDetectorAPI(MagicMock())
    assert factory.call_count == 1
    assert factory.call_args.args[0].onnx_path.endswith('det_yolo_v6_split.onnx')


class SyntheticRunner:
    input_infos = (TensorInfo('images', (1, 3, 32, 32), 'tensor(float)'),)
    output_infos = (TensorInfo('output0', (1, 16, 1), 'tensor(float)'),)
    providers = ('FakeExecutionProvider',)

    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        self.calls = 0

    def run(self, inputs):
        self.calls += 1
        return [self.rows.copy() if self.calls == 1 else np.empty((0, 6), dtype=np.float32)]

    def close(self):
        pass


@pytest.fixture
def synthetic_pipeline(monkeypatch):
    monkeypatch.setattr('services.tiled_yolo.run_yolo_nms',
                        lambda rows, **_kwargs: [rows.copy()])
    monkeypatch.setattr('services.tiled_yolo.restore_yolo_boxes',
                        lambda rows, *_args: rows.copy())


def api_with_rows(rows):
    api = object.__new__(DCFuseDetectorAPI)
    api.settings = MagicMock()
    api.detector = DCFuseDetector(
        SyntheticRunner(rows), 0.4, tiled_inference=True, tile_overlap=200,
    )
    return api


def ideal_rows(judge):
    counts = {
        0: judge.ways,
        1: 2 if judge.is_detect_lower_screw else 0,
        2: min(judge.metal_piece_counts) if judge.is_detect_metal_piece else 0,
        8: judge.ways * 2 if judge.is_detect_nut else 0,
        9: judge.ways * 2 if judge.is_detectscrew else 0,
        10: judge.ways if judge.is_small_screw else 0,
        11: 2 if judge.is_detect_upper_screw else 0,
    }
    rows = []
    for class_id, count in counts.items():
        for index in range(count):
            left, top = 10 + index * 3, 10 + class_id * 5
            rows.append([left, top, left + 1, top + 1, 0.9, class_id])
    return rows


@pytest.mark.parametrize('product_type', list(DCFuseDetectorAPI.SUPPORTED_TYPES))
def test_all_product_rules_work_after_tiled_detection(synthetic_pipeline, product_type):
    api = api_with_rows(ideal_rows(DCFuseDetectorAPI.SUPPORTED_TYPES[product_type]))
    result = api.detect(InputParamsBusiness(
        image=np.zeros((500, 500, 3), dtype=np.uint8), product_type=product_type,
    )).to_dict()
    assert api.detector.runner.calls == 4
    assert result['verdict'] == 'PASS' and result['status'] == 'true'
    assert all(item['verdict'] == 'PASS' for item in result['detailList'])
    assert all(len(item['coordinate']) == 8 for item in result['detailList'])
    assert all(0 <= value <= 1 for item in result['detailList'] for value in item['coordinate'])


def test_missing_label_not_removed_by_present_label(synthetic_pipeline):
    product = '五路有熔丝盒有磁环'
    rows = ideal_rows(DCFuseDetectorAPI.SUPPORTED_TYPES[product])
    screw = next(row for row in rows if row[5] == 9)
    rows.append([*screw[:4], 0.8, 5])
    api = api_with_rows(rows)
    result = api.detect(InputParamsBusiness(
        image=np.zeros((500, 500, 3), dtype=np.uint8), product_type=product,
    )).to_dict()
    assert result['verdict'] == 'FAIL' and result['status'] == 'false'
    assert any(item['scene'] == 'no_screw_1' for item in result['detailList'])


def test_empty_tiled_result_is_fail_not_execution_error(synthetic_pipeline):
    api = api_with_rows([])
    result = api.detect(InputParamsBusiness(
        image=np.zeros((500, 500, 3), dtype=np.uint8),
        product_type='五路有熔丝盒有磁环',
    )).to_dict()
    assert result['verdict'] == 'FAIL' and result['status'] == 'false'
    assert result['detailList'] == [] and not result['error_msg']


def test_execution_failure_propagates_instead_of_becoming_fail():
    api = api_with_rows([])
    api.detector.runner.run = MagicMock(side_effect=RuntimeError('model unavailable'))
    with pytest.raises(ModelInferenceError):
        api.detect(InputParamsBusiness(
            image=np.zeros((500, 500, 3), dtype=np.uint8),
            product_type='五路有熔丝盒有磁环',
        ))


def test_weak_overlapping_metal_is_removed_before_business_count(synthetic_pipeline):
    product = '五路有熔丝盒有磁环'
    rows = ideal_rows(DCFuseDetectorAPI.SUPPORTED_TYPES[product])
    metal = next(row for row in rows if row[5] == 2)
    # Intersection is only 10% of either box, below the previous thresholds.
    rows.append([metal[0] + 0.9, metal[1], metal[2] + 0.9, metal[3], 0.45, 2])
    api = api_with_rows(rows)
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    result = api.detect(InputParamsBusiness(image=image, product_type=product)).to_dict()
    assert result['verdict'] == 'PASS'
    assert sum(item['scene'] == 'metal_piece_4' for item in result['detailList']) == 4
