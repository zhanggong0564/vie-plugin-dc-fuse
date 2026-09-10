from unittest.mock import MagicMock, patch

import pytest

from services.inference import OnnxRuntimeOptions, RunnerSpec
from services.scenario_registry import scenario_registry


def test_scene_registers_with_scenario_registry():
    from vie_plugin_dc_fuse.business_logic import DCFuseDetectorAPI

    assert scenario_registry.snapshot()["dc_fuse"] is DCFuseDetectorAPI


def test_business_initialization_creates_and_injects_runner():
    from vie_plugin_dc_fuse.business_logic import DCFuseDetectorAPI

    settings = MagicMock()
    runner = MagicMock()
    with (
        patch(
            "vie_plugin_dc_fuse.business_logic.create_inference_runner",
            return_value=runner,
        ) as runner_factory,
        patch("vie_plugin_dc_fuse.business_logic.DCFuseDetector") as detector_class,
    ):
        api = DCFuseDetectorAPI(settings)

    runner_factory.assert_called_once_with(
        RunnerSpec(
            scenario="dc_fuse",
            onnx_path="./weights/dc_fuse/det_yolo_v6.onnx",
        ),
        OnnxRuntimeOptions.from_settings(settings),
    )
    detector_class.assert_called_once_with(runner, 0.6)
    assert api.detector is detector_class.return_value


def test_business_initialization_closes_runner_when_detector_creation_fails():
    from vie_plugin_dc_fuse.business_logic import DCFuseDetectorAPI

    runner = MagicMock()
    with (
        patch(
            "vie_plugin_dc_fuse.business_logic.create_inference_runner",
            return_value=runner,
        ),
        patch(
            "vie_plugin_dc_fuse.business_logic.DCFuseDetector",
            side_effect=RuntimeError("detector failed"),
        ),
        pytest.raises(Exception, match="dc_fuse 模型加载失败"),
    ):
        DCFuseDetectorAPI(MagicMock())

    runner.close.assert_called_once_with()


def test_business_close_is_idempotent():
    from vie_plugin_dc_fuse.business_logic import DCFuseDetectorAPI

    detector = MagicMock()
    with (
        patch("vie_plugin_dc_fuse.business_logic.create_inference_runner"),
        patch(
            "vie_plugin_dc_fuse.business_logic.DCFuseDetector",
            return_value=detector,
        ),
    ):
        api = DCFuseDetectorAPI(MagicMock())

    api.close()
    api.close()

    detector.close.assert_called_once_with()
