"""请求示例与插件实际解析、输入转换保持一致。"""

import asyncio
import copy

import numpy as np

from vie_plugin_dc_fuse.plugin import dc_fuse_router


def test_document_example_matches_request_parser():
    router = dc_fuse_router
    payload = copy.deepcopy(router.request_document_example)
    request = router.request_schema(payload)
    assert isinstance(request, router.request_document_model)
    assert request.model_dump(by_alias=True) == router.request_document_model.model_validate(payload).model_dump(by_alias=True)


def test_document_example_builds_inputs():
    router = dc_fuse_router
    request = router.request_schema(copy.deepcopy(router.request_document_example))
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    inputs = router.get_inputs(request, image)
    assert inputs.product_type == request.AICameraModel[0].AIParameterValue
    assert inputs.image is image


def test_document_example_legacy_compatibility():
    import pytest
    from pydantic import ValidationError

    payload = copy.deepcopy(dc_fuse_router.request_document_example)
    product_type = payload["AICameraModel"][0]["AIParameterValue"]
    payload["modelParams"]["product_model"] = product_type
    assert dc_fuse_router.request_schema(payload).product_type == product_type
    payload["modelParams"]["product_model"] = "冲突型号"
    with pytest.raises(ValidationError, match="新旧产品类型参数不一致"):
        dc_fuse_router.request_schema(payload)
    payload["AICameraModel"] = []
    assert dc_fuse_router.request_schema(payload).product_type == "冲突型号"
