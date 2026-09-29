"""直流熔丝检测业务逻辑：适配模板方法基类，型号判定收敛到 business_post_process(ctx)。

判定规则（ResultJudge）原样移植自旧框架 services/dc_fuse/business_logic.py，仅将
旧的 business_logic_post_process(result, product_type) 改写为新模板 business_post_process(ctx)：
裸返回 → 写 ctx.result（MoMResult），坐标输出像素值，归一化交给基类 normalize_hook。
"""

from collections import defaultdict
from copy import copy

from services.base import BusinessLogicBase
from services.inference import (
    OnnxRuntimeOptions,
    RunnerSpec,
    create_inference_runner,
)
from services.scenario_registry import scenario_registry
from schemas.data_base import MoMResult, DetectionItem, MessageType, InputParamsBusiness
from schemas.exceptions import ProductNotRegisteredError, ModelInferenceError
from schemas.inference_context import InferenceContext
from utils import vision_logger
from .dc_fuse_detect import DCFuseDetector


class ResultJudge:
    def __init__(
        self,
        ways=5,
        is_detect_metal_piece=True,
        is_detect_upper_screw=False,
        is_detect_lower_screw=False,
        is_detect_nut=False,
        is_detectscrew=True,
        is_small_screw=False,
        metal_piece_counts=(4,),
    ):
        self.ways = ways
        self.is_detect_metal_piece = is_detect_metal_piece
        self.is_detect_upper_screw = is_detect_upper_screw
        self.is_detect_lower_screw = is_detect_lower_screw
        self.is_detect_nut = is_detect_nut
        self.is_detectscrew = is_detectscrew
        self.is_small_screw = is_small_screw
        self.metal_piece_counts = frozenset(metal_piece_counts)

    def __call__(self, det_info, *, merged_crossbeam: bool = False):
        screw = det_info.get("screw_1", [])
        nut = det_info.get("nut_2", [])
        brass_plate = det_info.get("brass_plate_6", [])
        metal_piece = det_info.get("metal_piece_4", [])
        no_screw = det_info.get("no_screw_1", [])
        no_nut = det_info.get("no_nut2", [])
        upper_screw = det_info.get("upper_crossbeam_screw_9", [])
        lower_screw = det_info.get("lower_crossbeam_screw_10", [])
        no_upper_screw = det_info.get("no_upper_crossbeam_screw_9", [])
        no_lower_screw = det_info.get("no_lower_crossbeam_screw_10", [])
        small_screw = det_info.get("small_screw_8", [])
        no_small_screw = det_info.get("no_small_screw_8", [])
        results = {
            "screw": True,
            "nut": True,
            "metal_piece": True,
            "upper_screw": True,
            "lower_screw": True,
            "brass_plate": len(brass_plate) == self.ways,
            "small_screw": True,
        }
        if self.is_detectscrew:
            results["screw"] = len(screw) == self.ways * 2 and not no_screw
        if self.is_small_screw:
            results["small_screw"] = (
                len(small_screw) == self.ways and not no_small_screw
            )
        if self.is_detect_nut:
            results["nut"] = len(nut) == self.ways * 2 and not no_nut
        if self.is_detect_metal_piece:
            results["metal_piece"] = len(metal_piece) in self.metal_piece_counts
        if self.is_detect_upper_screw:
            results["upper_screw"] = len(upper_screw) == 2 and not no_upper_screw
        if self.is_detect_lower_screw:
            results["lower_screw"] = len(lower_screw) == 2 and not no_lower_screw
        if merged_crossbeam:
            # 仅同时检查上下横梁的已验证型号允许使用合并数量规则。
            if not (self.is_detect_upper_screw and self.is_detect_lower_screw):
                raise ValueError("merged crossbeam rule requires both beams")
            results.pop("upper_screw")
            results.pop("lower_screw")
            results["crossbeam_screw"] = (
                len(det_info.get("crossbeam_screw", [])) == 4
                and not det_info.get("no_crossbeam_screw")
            )
        return {
            key: value
            for key, value in results.items()
            if self._is_detection_enabled(key)
        }

    def _is_detection_enabled(self, key: str) -> bool:
        """检查指定检测项是否启用"""
        detection_map = {
            "screw": self.is_detectscrew,
            "nut": self.is_detect_nut,
            "metal_piece": self.is_detect_metal_piece,
            "upper_screw": self.is_detect_upper_screw,
            "lower_screw": self.is_detect_lower_screw,
            "brass_plate": True,
            "crossbeam_screw": self.is_detect_upper_screw and self.is_detect_lower_screw,
            "small_screw": self.is_small_screw,
        }
        return detection_map.get(key, False)


@scenario_registry.register("dc_fuse")
class DCFuseDetectorAPI(BusinessLogicBase):
    MERGED_PRODUCT_TYPE = "五路有熔丝盒有磁环"

    SUPPORTED_TYPES = {
        "五路有熔丝盒有磁环": ResultJudge(
            ways=5,
            is_detectscrew=True,
            is_small_screw=True,
            is_detect_metal_piece=True,
            is_detect_upper_screw=True,
            is_detect_lower_screw=True,
        ),
        "五路有熔丝盒无磁环": ResultJudge(
            ways=5,
            is_detectscrew=True,
            is_detect_nut=True,
            is_detect_metal_piece=True,
        ),
        "六路无熔丝盒无磁环": ResultJudge(
            ways=6,
            is_detectscrew=False,
            is_detect_metal_piece=True,
            is_detect_nut=True,
        ),
        "六路有熔丝盒无磁环": ResultJudge(
            ways=6,
            is_detectscrew=True,
            is_detect_nut=True,
            is_detect_metal_piece=True,
        ),
        "七路无熔丝盒无磁环": ResultJudge(ways=7, is_detectscrew=False, is_detect_nut=True),
        "七路有熔丝盒无磁环": ResultJudge(
            ways=7,
            is_detect_metal_piece=True,
            is_detectscrew=True,
            is_detect_nut=True,
            metal_piece_counts=(2,),
        ),
        "双层五路有熔丝盒无磁环": ResultJudge(
            ways=5,
            is_detectscrew=True,
            is_detect_metal_piece=True,
            is_detect_lower_screw=True,
            metal_piece_counts=(6,),
        ),
        "双层六路无熔丝盒无磁环": ResultJudge(
            ways=6,
            is_detectscrew=True,
            is_detect_metal_piece=True,
            is_detect_lower_screw=True,
            metal_piece_counts=(6,),
        ),
        "双层六路有熔丝盒无磁环": ResultJudge(
            ways=6,
            is_detectscrew=True,
            is_detect_metal_piece=True,
            is_detect_lower_screw=True,
            metal_piece_counts=(6,),
        ),
        "双层七路无熔丝盒无磁环": ResultJudge(
            ways=7,
            is_detectscrew=True,
            is_detect_metal_piece=True,
            is_detect_lower_screw=True,
            metal_piece_counts=(4, 6),
        ),
        "双层七路有熔丝盒无磁环": ResultJudge(
            ways=7,
            is_detectscrew=True,
            is_detect_metal_piece=True,
            is_detect_lower_screw=True,
            metal_piece_counts=(4, 6),
        ),
    }

    # 判定项 -> 该项对应的检测标签（含 no_ 前缀），用于回填 detailList，无每请求状态故置类属性
    label_mapping = {
        "crossbeam_screw": ["crossbeam_screw", "no_crossbeam_screw"],
        "screw": ["screw_1", "no_screw_1"],
        "nut": ["nut_2", "no_nut2"],
        "small_screw": ["small_screw_8", "no_small_screw_8"],
        "brass_plate": ["brass_plate_6"],
        "metal_piece": ["metal_piece_4"],
        "upper_screw": ["upper_crossbeam_screw_9", "no_upper_crossbeam_screw_9"],
        "lower_screw": ["lower_crossbeam_screw_10", "no_lower_crossbeam_screw_10"],
    }

    def _initialize_model(self, settings):
        from .config import DcFuseConfig

        cfg = DcFuseConfig()
        runner = None
        merged_runner = None
        self.merged_detector = None
        try:
            runner = create_inference_runner(
                RunnerSpec(scenario="dc_fuse", onnx_path=cfg.inference_model_path),
                OnnxRuntimeOptions.from_settings(settings),
            )
            if cfg.tiled_inference:
                self.detector = DCFuseDetector(
                    runner, cfg.inference_conf_threshold,
                    tiled_inference=True,
                    tile_overlap=cfg.tile_overlap,
                    class_conf_thresholds=cfg.tiled_class_conf_thresholds.model_dump(),
                    copper_max_aspect_ratio=cfg.tiled_copper_max_aspect_ratio,
                )
            else:
                self.detector = DCFuseDetector(runner, cfg.confThreshold)
            if cfg.tiled_inference and cfg.merged_inference:
                merged_runner = create_inference_runner(
                    RunnerSpec(scenario="dc_fuse", onnx_path=cfg.merged_model_path,
                               model_role="merged_crossbeam"),
                    OnnxRuntimeOptions.from_settings(settings),
                )
                thresholds = {
                    name: value for name, value in cfg.tiled_class_conf_thresholds.model_dump().items()
                    if name not in ("upper_crossbeam_screw_9", "lower_crossbeam_screw_10")
                }
                thresholds["crossbeam_screw"] = cfg.merged_crossbeam_conf_threshold
                self.merged_detector = DCFuseDetector(
                    merged_runner, cfg.tiled_conf_threshold, merged_classes=True,
                    tiled_inference=True, tile_overlap=cfg.tile_overlap,
                    class_conf_thresholds=thresholds,
                    copper_max_aspect_ratio=cfg.tiled_copper_max_aspect_ratio,
                )
        except Exception as e:
            if merged_runner is not None:
                merged_runner.close()
            if runner is not None:
                try:
                    runner.close()
                except Exception as close_error:
                    vision_logger.warning(
                        f"dc_fuse 初始化回滚清理失败: {close_error}"
                    )
            vision_logger.error(f"initialize model failed, error: {e}")
            raise ModelInferenceError(
                "dc_fuse 模型加载失败",
                scenario="dc_fuse",
                original_error=e,
            ) from e

    def detect(self, params: InputParamsBusiness) -> MoMResult:
        merged = getattr(self, "merged_detector", None)
        if merged is not None and params.product_type == self.MERGED_PRODUCT_TYPE:
            # 使用请求局部视图，绝不修改共享单例的detector，避免并发串用模型。
            pipeline = copy(self)
            pipeline.detector = merged
            return BusinessLogicBase.detect(pipeline, params)
        return super().detect(params)

    def close(self) -> None:
        try:
            super().close()
        finally:
            merged = getattr(self, "merged_detector", None)
            if merged is not None:
                self.merged_detector = None
                merged.close()

    def business_post_process(self, ctx: InferenceContext) -> None:
        product_type = ctx.product_type
        if product_type not in self.SUPPORTED_TYPES:
            raise ProductNotRegisteredError(
                f"产品型号 '{product_type}' 未在 dc_fuse SUPPORTED_TYPES 中注册",
                product_type=product_type,
                scenario="dc_fuse",
            )
        result = ctx.raw_result  # DetectResult
        result_judge = self.SUPPORTED_TYPES[product_type]
        det_info = defaultdict(list)
        for bbox, score, name in zip(
            result.boxes,
            result.scores,
            result.class_names,
        ):
            det_info[name].append({"bbox": bbox, "score": score})
        merged_crossbeam = getattr(getattr(self, "detector", None), "nc", 12) == 10
        judge_result = result_judge(det_info, merged_crossbeam=merged_crossbeam)
        # 坐标输出像素 xyxy，归一化由基类 normalize_hook 统一处理（NORMALIZE 默认 True）
        mom_result = MoMResult(status=True, message=MessageType.SUCCESS.value)
        for label, is_pass in judge_result.items():
            if not is_pass:
                mom_result.status = False
            for sub_label in self.label_mapping.get(label, []):
                for det in det_info.get(sub_label, []):
                    mom_result.detailList.append(
                        DetectionItem(
                            status=is_pass,
                            scene=sub_label,
                            coordinate=det["bbox"],
                            accuracy=det["score"],
                        )
                    )
        ctx.result = mom_result
