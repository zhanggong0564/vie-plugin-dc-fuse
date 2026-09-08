"""dc_fuse 插件单元测试：型号判定逻辑、business_post_process 契约、未注册型号异常。"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from schemas.exceptions import ProductNotRegisteredError
from schemas.inference_context import InferenceContext
from schemas.data_base import DetectResult
from vie_plugin_dc_fuse.business_logic import DCFuseDetectorAPI, ResultJudge


def _passing_detections(
    ways,
    *,
    metal_piece_count=4,
    crossbeam=False,
    lower_crossbeam=False,
):
    detections = {
        "screw_1": [{}] * (ways * 2),
        "nut_2": [{}] * (ways * 2),
        "brass_plate_6": [{}] * ways,
        "metal_piece_4": [{}] * metal_piece_count,
        "small_screw_8": [{}] * ways,
    }
    if crossbeam:
        detections["upper_crossbeam_screw_9"] = [{}] * 2
        detections["lower_crossbeam_screw_10"] = [{}] * 2
    elif lower_crossbeam:
        detections["lower_crossbeam_screw_10"] = [{}] * 2
    return detections


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


@pytest.mark.parametrize(
    ("product_type", "detections"),
    [
        (
            "五路有熔丝盒有磁环",
            _passing_detections(5, crossbeam=True),
        ),
        (
            "五路有熔丝盒无磁环",
            _passing_detections(5),
        ),
        (
            "六路无熔丝盒无磁环",
            _passing_detections(6),
        ),
        (
            "六路有熔丝盒无磁环",
            _passing_detections(6),
        ),
        (
            "七路无熔丝盒无磁环",
            _passing_detections(7),
        ),
        (
            "七路有熔丝盒无磁环",
            _passing_detections(7, metal_piece_count=2),
        ),
        (
            "双层五路有熔丝盒无磁环",
            _passing_detections(5, metal_piece_count=6, lower_crossbeam=True),
        ),
        (
            "双层六路无熔丝盒无磁环",
            _passing_detections(6, metal_piece_count=6, lower_crossbeam=True),
        ),
        (
            "双层六路有熔丝盒无磁环",
            _passing_detections(6, metal_piece_count=6, lower_crossbeam=True),
        ),
        (
            "双层七路无熔丝盒无磁环",
            _passing_detections(7, metal_piece_count=4, lower_crossbeam=True),
        ),
        (
            "双层七路有熔丝盒无磁环",
            _passing_detections(7, metal_piece_count=6, lower_crossbeam=True),
        ),
    ],
)
def test_registered_model_rules_accept_expected_counts(product_type, detections):
    judge = DCFuseDetectorAPI.SUPPORTED_TYPES[product_type]

    assert all(judge(detections).values())


@pytest.mark.parametrize(
    ("judge", "positive_label", "negative_label", "result_key"),
    [
        (ResultJudge(ways=5), "screw_1", "no_screw_1", "screw"),
        (
            ResultJudge(ways=5, is_detect_nut=True),
            "nut_2",
            "no_nut2",
            "nut",
        ),
        (
            ResultJudge(ways=5, is_small_screw=True),
            "small_screw_8",
            "no_small_screw_8",
            "small_screw",
        ),
    ],
)
def test_negative_label_rejects_an_otherwise_valid_count(
    judge,
    positive_label,
    negative_label,
    result_key,
):
    detections = _passing_detections(5)
    assert len(detections[positive_label]) > 0
    detections[negative_label] = [{}]

    assert judge(detections)[result_key] is False


def test_screw_count_must_match_even_without_negative_label():
    judge = ResultJudge(ways=5)
    detections = _passing_detections(5)
    detections["screw_1"] = [{}] * 9

    assert judge(detections)["screw"] is False


@pytest.mark.parametrize(
    ("missing_label", "result_key"),
    [
        ("no_upper_crossbeam_screw_9", "upper_screw"),
        ("no_lower_crossbeam_screw_10", "lower_screw"),
    ],
)
def test_crossbeam_screws_are_judged_independently(missing_label, result_key):
    judge = ResultJudge(
        ways=5,
        is_detect_upper_screw=True,
        is_detect_lower_screw=True,
    )
    detections = _passing_detections(5, crossbeam=True)
    detections[missing_label] = [{}]

    result = judge(detections)

    assert result[result_key] is False
    other_key = "lower_screw" if result_key == "upper_screw" else "upper_screw"
    assert result[other_key] is True


@pytest.mark.parametrize(
    "product_type",
    [
        "五路有熔丝盒有磁环",
        "五路有熔丝盒无磁环",
        "六路无熔丝盒无磁环",
        "六路有熔丝盒无磁环",
        "七路无熔丝盒无磁环",
    ],
)
def test_single_layer_models_require_exactly_four_metal_pieces(product_type):
    judge = DCFuseDetectorAPI.SUPPORTED_TYPES[product_type]

    assert judge(_passing_detections(judge.ways, metal_piece_count=4))["metal_piece"]
    assert not judge(_passing_detections(judge.ways, metal_piece_count=6))["metal_piece"]


@pytest.mark.parametrize("metal_piece_count", [4, 6])
def test_single_layer_seven_way_fuse_model_requires_two_metal_pieces(
    metal_piece_count,
):
    judge = DCFuseDetectorAPI.SUPPORTED_TYPES["七路有熔丝盒无磁环"]

    assert judge(_passing_detections(7, metal_piece_count=2))["metal_piece"]
    assert not judge(
        _passing_detections(7, metal_piece_count=metal_piece_count)
    )["metal_piece"]


@pytest.mark.parametrize("metal_piece_count", [4, 6])
@pytest.mark.parametrize(
    "product_type",
    [
        "双层七路无熔丝盒无磁环",
        "双层七路有熔丝盒无磁环",
    ],
)
def test_double_layer_seven_way_models_accept_four_or_six_metal_pieces(
    product_type,
    metal_piece_count,
):
    judge = DCFuseDetectorAPI.SUPPORTED_TYPES[product_type]
    detections = _passing_detections(
        7,
        metal_piece_count=metal_piece_count,
        lower_crossbeam=True,
    )

    assert all(judge(detections).values())


@pytest.mark.parametrize(
    ("product_type", "rejected_counts"),
    [
        ("双层五路有熔丝盒无磁环", (4, 5, 7)),
        ("双层六路无熔丝盒无磁环", (4, 5, 7)),
        ("双层六路有熔丝盒无磁环", (4, 5, 7)),
        ("双层七路无熔丝盒无磁环", (3, 5, 7)),
        ("双层七路有熔丝盒无磁环", (3, 5, 7)),
    ],
)
def test_double_layer_models_reject_unconfigured_metal_piece_counts(
    product_type,
    rejected_counts,
):
    judge = DCFuseDetectorAPI.SUPPORTED_TYPES[product_type]

    for count in rejected_counts:
        assert not judge(
            _passing_detections(judge.ways, metal_piece_count=count)
        )["metal_piece"]


@pytest.mark.parametrize(
    ("product_type", "ways"),
    [
        ("双层五路有熔丝盒无磁环", 5),
        ("双层六路无熔丝盒无磁环", 6),
        ("双层六路有熔丝盒无磁环", 6),
        ("双层七路无熔丝盒无磁环", 7),
        ("双层七路有熔丝盒无磁环", 7),
    ],
)
def test_double_layer_model_rules(product_type, ways):
    judge = DCFuseDetectorAPI.SUPPORTED_TYPES[product_type]
    expected_keys = {"screw", "metal_piece", "lower_screw", "brass_plate"}
    detections = _passing_detections(
        ways,
        metal_piece_count=next(iter(judge.metal_piece_counts)),
        lower_crossbeam=True,
    )

    assert set(judge(detections)) == expected_keys

    for label in ("brass_plate_6", "screw_1", "lower_crossbeam_screw_10"):
        too_few = {key: list(value) for key, value in detections.items()}
        too_few[label] = too_few[label][:-1]
        assert not all(judge(too_few).values())

        too_many = {key: list(value) for key, value in detections.items()}
        too_many[label].append({})
        assert not all(judge(too_many).values())

    missing = {key: list(value) for key, value in detections.items()}
    missing["no_screw_1"] = [{}]
    assert judge(missing)["screw"] is False

    missing = {key: list(value) for key, value in detections.items()}
    missing["no_lower_crossbeam_screw_10"] = [{}]
    assert judge(missing)["lower_screw"] is False

    ignored = {key: list(value) for key, value in detections.items()}
    ignored.update(
        {
            "nut_2": [{}] * 99,
            "no_nut2": [{}],
            "small_screw_8": [{}] * 99,
            "no_small_screw_8": [{}],
            "upper_crossbeam_screw_9": [{}] * 99,
            "no_upper_crossbeam_screw_9": [{}],
        }
    )
    assert all(judge(ignored).values())


def test_double_layer_five_way_with_magnetic_ring_is_not_registered(api):
    ctx = InferenceContext(
        image=np.zeros((10, 10, 3), np.uint8),
        h=10,
        w=10,
        product_type="双层五路有熔丝盒有磁环",
    )

    with pytest.raises(ProductNotRegisteredError):
        api.business_post_process(ctx)


@pytest.fixture
def api():
    """绕过真实模型加载，构造 DCFuseDetectorAPI。"""
    with (
        patch("vie_plugin_dc_fuse.business_logic.create_inference_runner"),
        patch("vie_plugin_dc_fuse.business_logic.DCFuseDetector"),
    ):
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
    assert set(out.keys()) == {
        "status",
        "verdict",
        "detailList",
        "error_msg",
        "message",
    }


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
