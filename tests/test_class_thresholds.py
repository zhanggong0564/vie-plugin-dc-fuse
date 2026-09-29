"""分类阈值筛选及铜排几何过滤的回归与配置契约。"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from schemas.data_base import InputParamsBusiness
from services.inference import TensorInfo
from vie_plugin_dc_fuse.business_logic import DCFuseDetectorAPI
from vie_plugin_dc_fuse.config import DcFuseConfig
from vie_plugin_dc_fuse.dc_fuse_detect import DCFuseDetector


class RawRunner:
    input_infos = (TensorInfo('images', (1, 3, 32, 32), 'tensor(float)'),)
    output_infos = (TensorInfo('output0', (1, 16, None), 'tensor(float)'),)
    providers = ('FakeExecutionProvider',)

    def __init__(self, rows):
        self.prediction = np.zeros((1, 16, len(rows)), dtype=np.float32)
        for index, (x1, y1, x2, y2, score, class_id) in enumerate(rows):
            self.prediction[0, :4, index] = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
            self.prediction[0, 4+class_id, index] = score

    def run(self, inputs):
        return [self.prediction.copy()]

    def close(self):
        pass


def model(rows, **kwargs):
    cfg = DcFuseConfig(_env_file=None)
    options = dict(tiled_inference=True,
                   class_conf_thresholds=cfg.tiled_class_conf_thresholds.model_dump(),
                   copper_max_aspect_ratio=cfg.tiled_copper_max_aspect_ratio)
    options.update(kwargs)
    return DCFuseDetector(RawRunner(rows), 0.4, **options)


def test_default_class_thresholds():
    cfg = DcFuseConfig(_env_file=None)
    assert cfg.tiled_class_conf_thresholds.model_dump() == {
        'screw_1': 0.75, 'brass_plate_6': 0.75, 'small_screw_8': 0.70,
        'metal_piece_4': 0.45, 'upper_crossbeam_screw_9': 0.40,
        'lower_crossbeam_screw_10': 0.40,
    }
    assert cfg.tiled_copper_max_aspect_ratio == 1.0


def test_nested_environment_overrides_preserve_other_class_defaults(monkeypatch):
    monkeypatch.setenv('DC_FUSE_TILED_CLASS_CONF_THRESHOLDS', '{"brass_plate_6":0.8}')
    monkeypatch.setenv('DC_FUSE_TILED_CLASS_CONF_THRESHOLDS__SCREW_1', '0.76')
    monkeypatch.setenv('DC_FUSE_TILED_COPPER_MAX_ASPECT_RATIO', '0.9')
    cfg = DcFuseConfig(_env_file=None)
    assert cfg.tiled_class_conf_thresholds.brass_plate_6 == 0.8
    assert cfg.tiled_class_conf_thresholds.screw_1 == 0.76
    assert cfg.tiled_class_conf_thresholds.small_screw_8 == 0.70
    assert cfg.tiled_copper_max_aspect_ratio == 0.9
    explicit = DcFuseConfig(_env_file=None, tiled_class_conf_thresholds={'screw_1': 0.77})
    assert explicit.tiled_class_conf_thresholds.screw_1 == 0.77


@pytest.mark.parametrize('value', [-.01, 1.01, float('nan'), float('inf'), 'invalid'])
def test_invalid_class_threshold_rejected_at_configuration_load(value):
    with pytest.raises(ValidationError):
        DcFuseConfig(_env_file=None, tiled_class_conf_thresholds={'screw_1': value})


def test_unknown_config_class_rejected():
    with pytest.raises(ValidationError):
        DcFuseConfig(_env_file=None, tiled_class_conf_thresholds={'screw_typo': .7})


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), 'invalid'])
def test_invalid_copper_ratio_rejected_at_configuration_load(value):
    with pytest.raises(ValidationError):
        DcFuseConfig(_env_file=None, tiled_copper_max_aspect_ratio=value)


@pytest.mark.parametrize('class_id,threshold', [
    (0, .75), (1, .4), (2, .45), (3, .4), (4, .4), (5, .4),
    (6, .4), (7, .4), (8, .4), (9, .75), (10, .7), (11, .4),
])
@pytest.mark.parametrize('offset,kept', [(-.01, False), (0, False), (.01, True)])
def test_per_class_confidence_and_threshold_boundary(class_id, threshold, offset, kept):
    detector = model([[2, 2, 5, 8, threshold+offset, class_id]])
    result = detector.infer(np.zeros((32, 32, 3), dtype=np.uint8))
    assert result.class_ids == ([class_id] if kept else [])


def test_rejected_best_class_is_not_reclassified_as_second_best():
    detector = model([[2, 2, 5, 8, .74, 9]])
    detector.runner.prediction[0, 4+11, 0] = .6
    assert detector.infer(np.zeros((32, 32, 3), dtype=np.uint8)).boxes == []


def test_pre_nms_class_filter_preserves_raw_outputs_and_surviving_small_box():
    detector = model([[1, 1, 12, 12, .74, 9], [3, 3, 7, 7, .8, 9]])
    raw = detector.runner.prediction.copy()
    _, meta = detector.preprocess(np.zeros((32, 32, 3), dtype=np.uint8))
    result = detector.post_process([[raw]], meta)
    np.testing.assert_array_equal(raw, detector.runner.prediction)
    assert result.class_ids == [9]
    assert result.boxes == [[3, 3, 7, 7]]


def test_low_threshold_class_not_removed_by_high_global_prefilter():
    detector = DCFuseDetector(RawRunner([[2, 2, 5, 8, .41, 11]]), .6,
                             tiled_inference=True, class_conf_thresholds={'upper_crossbeam_screw_9': .4})
    assert detector.confThreshold == .4
    assert detector.infer(np.zeros((32, 32, 3), dtype=np.uint8)).class_ids == [11]


def test_copper_ratio_filters_only_copper_and_keeps_parallel_fields_aligned():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    detector = model([[1, 1, 7, 4, .9, 0], [8, 1, 11, 7, .8, 0],
                      [12, 1, 15, 4, .85, 0], [16, 1, 22, 4, .5, 5]])
    result = detector.infer(image)
    assert sorted(result.class_ids) == [0, 0, 5]
    assert len(result.boxes) == len(result.scores) == len(result.class_ids) == len(result.class_names) == 3
    assert result.ori_img is image
    for box, name, score in zip(result.boxes, result.class_names, result.scores):
        if name == 'brass_plate_6':
            assert (box[2]-box[0])/(box[3]-box[1]) <= 1
            assert score > .75
        else:
            assert name == 'no_screw_1' and score == .5


def test_original_mode_keeps_legacy_confidence_and_geometry():
    detector = model([[1, 1, 7, 4, .65, 0]], tiled_inference=False)
    assert detector.infer(np.zeros((32, 32, 3), dtype=np.uint8)).class_names == ['brass_plate_6']


def test_missing_label_still_fails_with_correct_positive_counts():
    # 五路有熔丝盒有磁环：所有正常件数量正确，再加低分缺件标签。
    rows = []
    for class_id, count in [(0, 5), (9, 10), (2, 4), (10, 5), (11, 2), (1, 2)]:
        for index in range(count):
            left = 1+index*2
            top = {0: 1, 9: 5, 2: 9, 10: 13, 11: 17, 1: 21}[class_id]
            rows.append([left, top, left+1, top+2, .9, class_id])
    rows.append([28, 5, 29, 7, .41, 5])
    api = object.__new__(DCFuseDetectorAPI)
    api.settings = MagicMock()
    api.detector = model(rows)
    result = api.detect(InputParamsBusiness(image=np.zeros((32,32,3), dtype=np.uint8),
                                          product_type='五路有熔丝盒有磁环')).to_dict()
    assert result['verdict'] == 'FAIL' and result['status'] == 'false'
    assert result['message'] == '检测成功' and not result['error_msg']
    assert any(item['scene'] == 'no_screw_1' for item in result['detailList'])
    assert sum(item['scene'] == 'screw_1' for item in result['detailList']) == 10
    assert all(item['verdict'] == 'PASS' for item in result['detailList']
               if item['scene'] not in ('screw_1', 'no_screw_1'))


def test_original_api_initialization_does_not_enable_class_filters(monkeypatch):
    monkeypatch.setenv('DC_FUSE_TILED_INFERENCE', 'false')
    with (patch('vie_plugin_dc_fuse.business_logic.create_inference_runner') as factory,
          patch('vie_plugin_dc_fuse.business_logic.DCFuseDetector') as detector):
        DCFuseDetectorAPI(MagicMock())
    detector.assert_called_once_with(factory.return_value, .6)


@pytest.mark.parametrize('product_type', list(DCFuseDetectorAPI.SUPPORTED_TYPES))
def test_configured_class_thresholds_keep_all_registered_product_rules(product_type):
    judge = DCFuseDetectorAPI.SUPPORTED_TYPES[product_type]
    counts = {
        0: judge.ways,
        1: 2 if judge.is_detect_lower_screw else 0,
        2: min(judge.metal_piece_counts) if judge.is_detect_metal_piece else 0,
        8: judge.ways*2 if judge.is_detect_nut else 0,
        9: judge.ways*2 if judge.is_detectscrew else 0,
        10: judge.ways if judge.is_small_screw else 0,
        11: 2 if judge.is_detect_upper_screw else 0,
    }
    rows = []
    for class_id, count in counts.items():
        for index in range(count):
            left, top = 1+index*2, 1+class_id*2
            rows.append([left, top, left+1, top+2, .9, class_id])
    api = object.__new__(DCFuseDetectorAPI)
    api.settings = MagicMock()
    api.detector = model(rows)
    result = api.detect(InputParamsBusiness(
        image=np.zeros((32, 32, 3), dtype=np.uint8), product_type=product_type,
    )).to_dict()
    assert result['verdict'] == 'PASS' and result['status'] == 'true'
    assert all(item['verdict'] == 'PASS' for item in result['detailList'])
    assert all(len(item['coordinate']) == 8 for item in result['detailList'])
