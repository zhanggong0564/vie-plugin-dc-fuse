"""响应文档示例与 mock 检测结果经过真实业务后处理的契约。"""
from types import SimpleNamespace

import numpy as np
import pytest

from schemas.data_base import DetectResult
from vie_plugin_dc_fuse.business_logic import DCFuseDetectorAPI
from vie_plugin_dc_fuse.response_docs import RESPONSE_EXAMPLES


@pytest.mark.parametrize("name", list(RESPONSE_EXAMPLES))
def test_documented_response_matches_business_postprocess(name):
    expected = RESPONSE_EXAMPLES[name]["result"]
    detections = [item for item in expected["detailList"] if item["coordinate"]]
    boxes = [[round(item["coordinate"][i] * 1000) for i in (0, 1, 4, 5)] for item in detections]
    raw = DetectResult(boxes=boxes, scores=[item["accuracy"] for item in detections],
                       class_names=[item["scene"] for item in detections])
    ctx = SimpleNamespace(raw_result=raw, product_type='六路无熔丝盒无磁环', image=np.zeros((1000, 1000, 3), np.uint8), w=1000, h=1000)
    api = object.__new__(DCFuseDetectorAPI)
    api.business_post_process(ctx)
    api.normalize_hook(ctx)
    actual = ctx.result.to_dict()
    assert actual["verdict"] == expected["verdict"]
    assert actual["status"] == expected["status"]
    assert actual["message"] == expected["message"]
    assert len(actual["detailList"]) == len(expected["detailList"])
    for item, documented in zip(actual["detailList"], expected["detailList"]):
        assert item["coordinate"] == pytest.approx(documented["coordinate"])
        assert {k: v for k, v in item.items() if k != "coordinate"} == {k: v for k, v in documented.items() if k != "coordinate"}
