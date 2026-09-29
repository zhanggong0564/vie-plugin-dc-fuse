from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import SettingsConfigDict

from services.base import SceneSettings


ConfidenceThreshold = Annotated[float, Field(ge=0, le=1)]


class DcFuseClassThresholds(BaseModel):
    """切图模型的正常件阈值；未列出的类别使用切图默认阈值。"""

    model_config = ConfigDict(extra="forbid")

    screw_1: ConfidenceThreshold = 0.75
    brass_plate_6: ConfidenceThreshold = 0.75
    small_screw_8: ConfidenceThreshold = 0.70
    metal_piece_4: ConfidenceThreshold = 0.45
    upper_crossbeam_screw_9: ConfidenceThreshold = 0.40
    lower_crossbeam_screw_10: ConfidenceThreshold = 0.40


class DcFuseConfig(SceneSettings):
    """直流熔丝场景配置（原 config/dc_fuse_confg.py，迁入插件自描述）。"""

    model_config = SettingsConfigDict(
        env_prefix="DC_FUSE_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
    )

    model_path: str = "./weights/dc_fuse/det_yolo_v6.onnx"
    conf_threshold: float = Field(default=0.6, ge=0, le=1)

    tiled_inference: bool = True
    tiled_model_path: str = Field(
        default="./weights/dc_fuse/det_yolo_v6_split.onnx", min_length=1,
    )
    tiled_conf_threshold: float = Field(default=0.4, ge=0, le=1)
    tile_overlap: int = Field(default=200, ge=0)
    tiled_class_conf_thresholds: DcFuseClassThresholds = Field(
        default_factory=DcFuseClassThresholds,
    )
    tiled_copper_max_aspect_ratio: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    # 已验证型号使用10类合并横梁模型；其他型号保留12类模型。
    merged_inference: bool = True
    merged_model_path: str = Field(
        default="./weights/dc_fuse/det_yolo_v6.2_split.onnx", min_length=1,
    )
    merged_crossbeam_conf_threshold: float = Field(default=0.4, ge=0, le=1)

    @property
    def inference_model_path(self) -> str:
        return self.tiled_model_path if self.tiled_inference else self.model_path

    @property
    def inference_conf_threshold(self) -> float:
        return self.tiled_conf_threshold if self.tiled_inference else self.conf_threshold

    @property
    def confThreshold(self) -> float:
        """Compatibility alias for the legacy scene configuration."""

        return self.conf_threshold
