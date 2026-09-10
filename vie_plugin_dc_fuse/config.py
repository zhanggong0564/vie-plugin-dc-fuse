from pydantic import Field
from pydantic_settings import SettingsConfigDict

from services.base import SceneSettings


class DcFuseConfig(SceneSettings):
    """直流熔丝场景配置（原 config/dc_fuse_confg.py，迁入插件自描述）。"""

    model_config = SettingsConfigDict(
        env_prefix="DC_FUSE_",
        env_file=".env",
        extra="ignore",
    )

    model_path: str = "./weights/dc_fuse/det_yolo_v6.onnx"
    conf_threshold: float = Field(default=0.6, ge=0, le=1)

    @property
    def confThreshold(self) -> float:
        """Compatibility alias for the legacy scene configuration."""

        return self.conf_threshold
