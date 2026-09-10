import pytest
from pydantic import ValidationError

from vie_plugin_dc_fuse.config import DcFuseConfig


def test_config_reads_prefixed_environment(monkeypatch):
    monkeypatch.setenv("DC_FUSE_CONF_THRESHOLD", "0.7")
    assert DcFuseConfig().conf_threshold == 0.7


def test_config_uses_v6_model_by_default():
    assert DcFuseConfig().model_path == "./weights/dc_fuse/det_yolo_v6.onnx"


def test_config_rejects_invalid_threshold(monkeypatch):
    monkeypatch.setenv("DC_FUSE_CONF_THRESHOLD", "1.1")
    with pytest.raises(ValidationError):
        DcFuseConfig()
