"""合并横梁模型的请求隔离、数量/缺件判定与错误契约。"""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from schemas.data_base import InputParamsBusiness
from schemas.exceptions import ModelInferenceError
from services.inference import TensorInfo
from vie_plugin_dc_fuse.business_logic import DCFuseDetectorAPI
from vie_plugin_dc_fuse.config import DcFuseConfig


class Runner:
    input_infos = (TensorInfo('images', (1, 3, 32, 32), 'tensor(float)'),)
    providers = ('FakeExecutionProvider',)

    def __init__(self, nc, rows):
        self.closed = 0
        self.output_infos = (TensorInfo('output0', (1, 4+nc, None), 'tensor(float)'),)
        self.prediction = np.zeros((1, 4+nc, len(rows)), dtype=np.float32)
        for i, (cid, x, y) in enumerate(rows):
            self.prediction[0, :4, i] = [x+.5, y+1, 1, 2]
            self.prediction[0, 4+cid, i] = .9

    def run(self, inputs):
        return [self.prediction.copy()]

    def close(self):
        self.closed += 1


def positive_rows(beam_count=4):
    return [(cid, 1+i*2, y) for cid, count, y in
            [(0, 5, 1), (8, 10, 5), (9, 5, 9), (2, 4, 13), (1, beam_count, 17)]
            for i in range(count)]


def api_with(rows, monkeypatch):
    monkeypatch.setenv('DC_FUSE_TILED_INFERENCE', 'true')
    monkeypatch.setenv('DC_FUSE_MERGED_INFERENCE', 'true')
    legacy_rows = [(cid, 1+i*2, y) for cid, count, y in
                   [(0, 5, 1), (9, 10, 5), (2, 4, 9), (8, 10, 13)] for i in range(count)]
    old, new = Runner(12, legacy_rows), Runner(10, rows)
    with patch('vie_plugin_dc_fuse.business_logic.create_inference_runner', side_effect=[old, new]):
        api = DCFuseDetectorAPI(MagicMock())
    return api, old, new


def request(product=DCFuseDetectorAPI.MERGED_PRODUCT_TYPE):
    return InputParamsBusiness(image=np.zeros((32,32,3), dtype=np.uint8), product_type=product)


@pytest.mark.parametrize('beam_count,expected', [(0, 'FAIL'), (3, 'FAIL'), (4, 'PASS'), (5, 'FAIL')])
def test_merged_beam_count_and_response_contract(monkeypatch, beam_count, expected):
    api, old, new = api_with(positive_rows(beam_count), monkeypatch)
    try:
        result = api.detect(request()).to_dict()
        assert result['verdict'] == expected
        assert result['status'] == ('true' if expected == 'PASS' else 'false')
        assert not result['error_msg']
        assert sum(d['scene']=='crossbeam_screw' for d in result['detailList']) == beam_count
        assert not any(d['scene'].startswith(('upper_', 'lower_')) for d in result['detailList'])
        assert all(len(d['coordinate'])==8 and all(0<=v<=1 for v in d['coordinate']) for d in result['detailList'])
    finally:
        api.close()
    assert old.closed == new.closed == 1


def test_missing_label_fails_even_with_four_positive_beams(monkeypatch):
    api, _, _ = api_with(positive_rows()+[(3, 27, 22)], monkeypatch)
    try:
        result = api.detect(request()).to_dict()
        assert result['verdict'] == 'FAIL'
        assert all(d['verdict']=='FAIL' for d in result['detailList']
                   if d['scene'] in ('crossbeam_screw', 'no_crossbeam_screw'))
        assert any(d['scene']=='no_crossbeam_screw' for d in result['detailList'])
    finally:
        api.close()


def test_empty_merged_result_is_fail(monkeypatch):
    api, _, _ = api_with([], monkeypatch)
    try:
        result=api.detect(request()).to_dict()
        assert result['verdict']=='FAIL' and result['detailList']==[] and not result['error_msg']
    finally:
        api.close()


def test_parallel_requests_keep_other_model_and_shared_detector(monkeypatch):
    api, old, new = api_with(positive_rows(), monkeypatch)
    legacy = api.detector
    params=[request(), request('五路有熔丝盒无磁环')]*10
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(api.detect, params))
        assert api.detector is legacy and api.detector.nc==12 and api.merged_detector.nc==10
        assert all(r.verdict.value=='PASS' for r in results)
        assert all(any(d.scene=='crossbeam_screw' for d in r.detailList)==(i%2==0)
                   for i,r in enumerate(results))
    finally:
        api.close()
        api.close()
    assert old.closed == new.closed == 1


def test_wrong_class_output_is_execution_error(monkeypatch):
    api, _, new = api_with(positive_rows(), monkeypatch)
    new.prediction=np.zeros((1,16,1), dtype=np.float32)
    try:
        with pytest.raises(ModelInferenceError):
            api.detect(request())
    finally:
        api.close()


def test_merged_init_failure_closes_both_runners(monkeypatch):
    monkeypatch.setenv('DC_FUSE_TILED_INFERENCE','true')
    monkeypatch.setenv('DC_FUSE_MERGED_INFERENCE','true')
    old,new=Runner(12,[]),Runner(10,[])
    with (patch('vie_plugin_dc_fuse.business_logic.create_inference_runner', side_effect=[old,new]),
          patch('vie_plugin_dc_fuse.business_logic.DCFuseDetector', side_effect=[MagicMock(),RuntimeError('bad merged model')]),
          pytest.raises(ModelInferenceError)):
        DCFuseDetectorAPI(MagicMock())
    assert old.closed==new.closed==1


@pytest.mark.parametrize('value', [-.1,1.1,float('nan'),'bad'])
def test_invalid_merged_threshold_rejected(value):
    with pytest.raises(ValidationError):
        DcFuseConfig(_env_file=None, merged_crossbeam_conf_threshold=value)


def test_merged_default_and_disable(monkeypatch):
    cfg=DcFuseConfig(_env_file=None)
    assert cfg.merged_inference and cfg.merged_model_path.endswith('det_yolo_v6.2_split.onnx')
    monkeypatch.setenv('DC_FUSE_MERGED_INFERENCE','false')
    assert not DcFuseConfig(_env_file=None).merged_inference
