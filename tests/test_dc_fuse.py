"""dc_fuse 插件单元测试：型号判定逻辑、business_post_process 契约、未注册型号异常。"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from schemas.exceptions import ProductNotRegisteredError
from schemas.inference_context import InferenceContext
from schemas.data_base import DetectResult
from vie_plugin_dc_fuse.business_logic import ResultJudge


def test_result_judge_flags_missing_items():
    # 七路无熔丝盒无磁环：启用 nut/metal_piece/brass_plate，未启用 screw
    judge = ResultJudge(ways=7, is_detectscrew=False, is_detect_nut=True)
    det_info = {
        "nut_2": [{}] * 14,         # 14 == ways*2 → 通过
        "metal_piece_4": [{}] * 4,  # 4 ∈ {4,6} → 通过
        "brass_plate_6": [{}] * 3,  # 3 != ways(7) → 不通过
    }
    res = judge(det_info)
    assert res["brass_plate"] is False
    assert res["nut"] is True
    assert res["metal_piece"] is True
    assert "screw" not in res  # is_detectscrew=False，该项未启用


@pytest.fixture
def api():
    """绕过真实模型加载，构造 DCFuseDetectorAPI。"""
    with patch("vie_plugin_dc_fuse.business_logic.DCFuseDetector"):
        from vie_plugin_dc_fuse.business_logic import DCFuseDetectorAPI
        yield DCFuseDetectorAPI(MagicMock())


def test_unregistered_product_raises(api):
    ctx = InferenceContext(image=np.zeros((10, 10, 3), np.uint8), h=10, w=10, product_type="不存在型号")
    with pytest.raises(ProductNotRegisteredError) as ei:
        api.business_post_process(ctx)
    assert ei.value.context.get("scenario") == "dc_fuse"


def test_business_post_process_builds_mom(api):
    ctx = InferenceContext(image=np.zeros((10, 10, 3), np.uint8), h=10, w=10,
                           product_type="七路无熔丝盒无磁环")
    # 全部满足：14 nut + 4 metal_piece + 7 brass_plate
    n = 14 + 4 + 7
    ctx.raw_result = DetectResult(
        boxes=[[1, 1, 2, 2]] * n,
        scores=[0.9] * n,
        class_ids=[0] * n,
        class_names=["nut_2"] * 14 + ["metal_piece_4"] * 4 + ["brass_plate_6"] * 7,
    )
    api.business_post_process(ctx)
    # 统一契约：ctx.result 为 MoMResult 对象
    assert ctx.result.status is True
    assert {it.scene for it in ctx.result.detailList} == {"nut_2", "metal_piece_4", "brass_plate_6"}
    assert len(ctx.result.detailList) == n
    # to_dict 输出形状与框架一致
    out = ctx.result.to_dict()
    assert out["status"] == "true"
    assert set(out.keys()) == {"status", "detailList", "error_msg", "message"}


def test_business_post_process_marks_fail(api):
    ctx = InferenceContext(image=np.zeros((10, 10, 3), np.uint8), h=10, w=10,
                           product_type="七路无熔丝盒无磁环")
    # brass_plate 不足 → 整体 status=False
    ctx.raw_result = DetectResult(
        boxes=[[1, 1, 2, 2]] * 3,
        scores=[0.9] * 3,
        class_ids=[0] * 3,
        class_names=["brass_plate_6"] * 3,
    )
    api.business_post_process(ctx)
    assert ctx.result.status is False
