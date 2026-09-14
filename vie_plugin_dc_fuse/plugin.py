"""Entry point: register the scene and expose ``dc_fuse_router``."""

import numpy as np

from routers.base_router import BaseRouter
from schemas.inspection import InspectionVerdict
from .response_docs import RESPONSE_EXAMPLES, RESPONSE_NOTES
from schemas.data_base import InputParamsBusiness
from .schemas import DCFuseRequest
from . import business_logic  # noqa: F401  触发 ScenarioRegistry 注册


class DCFuseRouter(BaseRouter):
    response_document_verdicts = (InspectionVerdict.PASS, InspectionVerdict.FAIL)
    response_document_notes = RESPONSE_NOTES
    response_document_examples = RESPONSE_EXAMPLES
    request_document_model = DCFuseRequest
    request_document_example = {
        "product": "直流熔丝", "type": "物料号",
        "modelParams": {"guide_line": [], "example_images": []},
        "AICameraModel": [{
            "Id": "registration-id", "Version": 1, "ModelFile": None,
            "AIParameterName": "产品类型", "AIParameterValue": "六路无熔丝盒无磁环",
        }],
    }
    request_document_notes = (
        "产品类型从 AICameraModel 中 AIParameterName=产品类型 的非空 AIParameterValue 获取。"
        "允许不同版本重复同一产品类型；无新参数时使用兼容字段 modelParams.product_model。"
        "新参数只有一个型号时，旧字段若非空必须与其一致；新参数有多个不同型号时，"
        "必须用旧字段指定其中一个型号，否则拒绝请求。新旧参数均缺失时也拒绝请求。"
    )

    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        super().__init__(router_name, api_path, summary, description, detector_type, tag=tag)

    def request_schema(self, json_dict):
        return DCFuseRequest(**json_dict)

    @staticmethod
    def _extract_product_type(request_params):
        # 推理与数据回流共用经 Schema 校验的产品类型。
        return request_params.product_type

    def get_inputs(self, request_params: DCFuseRequest, image: np.ndarray):
        return InputParamsBusiness(image=image, product_type=request_params.product_type)


dc_fuse_router = DCFuseRouter(
    router_name="dc_fuse_router",
    api_path="/dcfuse_detect",
    summary="直流熔丝检测接口",
    description="根据输入的图像和产品型号，返回直流熔丝检测结果",
    detector_type="dc_fuse",
    tag="直流熔丝检测",
)
