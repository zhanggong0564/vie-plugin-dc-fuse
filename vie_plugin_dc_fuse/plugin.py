"""entry_point 模块：导入 business_logic 触发工厂注册，并暴露 dc_fuse_router。"""

import numpy as np

from routers.base_router import BaseRouter
from schemas.data_base import InputParamsBusiness
from .schemas import DCFuseRequest
from . import business_logic  # noqa: F401  导入即触发 @detection_factory.register("dc_fuse")


class DCFuseRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        super().__init__(router_name, api_path, summary, description, detector_type, tag=tag)

    def request_schema(self, json_dict):
        return DCFuseRequest(**json_dict)

    @staticmethod
    def _extract_product_type(request_params):
        # 本场景型号字段名为 product_model，重写基类默认的 product_type 提取，
        # 使数据回流按型号分目录而非落到 _unknown_model。
        model_params = getattr(request_params, "modelParams", None)
        return getattr(model_params, "product_model", None) if model_params else None

    def get_inputs(self, request_params: DCFuseRequest, image: np.ndarray):
        product_model = request_params.modelParams.product_model
        return InputParamsBusiness(image=image, product_type=product_model)


dc_fuse_router = DCFuseRouter(
    router_name="dc_fuse_router",
    api_path="/dcfuse_detect",
    summary="直流熔丝检测接口",
    description="根据输入的图像和产品型号，返回直流熔丝检测结果",
    detector_type="dc_fuse",
    tag="直流熔丝检测",
)
