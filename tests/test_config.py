import pytest
from pydantic import ValidationError

from vie_plugin_dc_fuse.config import DcFuseConfig


def test_config_reads_prefixed_environment(monkeypatch):
    monkeypatch.setenv("DC_FUSE_CONF_THRESHOLD", "0.7")
    assert DcFuseConfig().conf_threshold == 0.7


def test_config_uses_split_model_by_default(monkeypatch):
    monkeypatch.delenv("DC_FUSE_TILED_INFERENCE", raising=False)
    monkeypatch.delenv("DC_FUSE_TILED_MODEL_PATH", raising=False)
    cfg = DcFuseConfig(_env_file=None)
    assert cfg.tiled_inference is True
    assert cfg.inference_model_path == "./weights/dc_fuse/det_yolo_v6_split.onnx"
    assert cfg.inference_conf_threshold == 0.4


def test_config_can_explicitly_restore_whole_image_mode(monkeypatch):
    monkeypatch.setenv("DC_FUSE_TILED_INFERENCE", "false")
    cfg = DcFuseConfig(_env_file=None)
    assert cfg.tiled_inference is False
    assert cfg.inference_model_path == "./weights/dc_fuse/det_yolo_v6.onnx"
    assert cfg.inference_conf_threshold == 0.6


def test_config_rejects_invalid_threshold(monkeypatch):
    monkeypatch.setenv("DC_FUSE_CONF_THRESHOLD", "1.1")
    with pytest.raises(ValidationError):
        DcFuseConfig()
