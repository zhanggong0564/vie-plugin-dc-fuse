"""直流熔丝检测器。"""

from collections.abc import Mapping
from math import isfinite

import numpy as np

from schemas.data_base import DetectResult
from schemas.inference_context import PreprocMeta
from services.inference import InferenceRunner
from services.tiled_yolo import TiledPreprocMeta, TiledYoloInfer


class DCFuseDetector(TiledYoloInfer):
    def __init__(
        self,
        runner: InferenceRunner,
        confThreshold=0.5,
        nmsThreshold=0.5,
        task="det",
        *,
        tiled_inference: bool = False,
        tile_overlap: int = 200,
        class_conf_thresholds: Mapping[str, float] | None = None,
        copper_max_aspect_ratio: float | None = None,
    ):
        super().__init__(
            nc=12,
            runner=runner,
            confThreshold=confThreshold,
            nmsThreshold=nmsThreshold,
            task=task,
            tiled_inference=tiled_inference,
            tile_overlap=tile_overlap,
        )
        self.id2name = {
            0: "brass_plate_6",
            1: "lower_crossbeam_screw_10",
            2: "metal_piece_4",
            3: "no_lower_crossbeam_screw_10",
            4: "no_nut2",
            5: "no_screw_1",
            6: "no_small_screw_8",
            7: "no_upper_crossbeam_screw_9",
            8: "nut_2",
            9: "screw_1",
            10: "small_screw_8",
            11: "upper_crossbeam_screw_9",
        }

        self.class_conf_thresholds = dict(class_conf_thresholds or {})
        unknown = self.class_conf_thresholds.keys() - self.id2name.values()
        if unknown:
            raise ValueError(f"unknown dc_fuse classes: {sorted(unknown)}")
        if any(not isfinite(value) or not 0 <= value <= 1
               for value in self.class_conf_thresholds.values()):
            raise ValueError("class confidence thresholds must be finite and within [0, 1]")
        if copper_max_aspect_ratio is not None and (
            not isfinite(copper_max_aspect_ratio) or copper_max_aspect_ratio <= 0
        ):
            raise ValueError("copper maximum aspect ratio must be positive and finite")
        self.copper_max_aspect_ratio = copper_max_aspect_ratio
        self._class_thresholds = np.asarray([
            self.class_conf_thresholds.get(self.id2name[index], confThreshold)
            for index in range(self.nc)
        ])
        if tiled_inference and self.class_conf_thresholds:
            # NMS 的统一预筛阈值不得提前删掉阈值较低的类别。
            self.confThreshold = float(self._class_thresholds.min())

    def _filter_class_confidence(self, prediction: np.ndarray) -> np.ndarray:
        """按原始最高分类筛选；不将被拒绝的框降级为第二类别。"""
        if prediction.ndim != 3 or prediction.shape[1] != 4 + self.nc:
            raise ValueError("dc_fuse requires YOLO box outputs with 12 classes")
        filtered = prediction.copy()
        scores = filtered[:, 4:, :]
        best_class = scores.argmax(axis=1)
        rejected = scores.max(axis=1) <= self._class_thresholds[best_class]
        scores *= (~rejected)[:, None, :]
        return filtered

    def post_process(
        self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        meta: PreprocMeta | TiledPreprocMeta,
    ) -> DetectResult:
        if not self.tiled_inference:
            return super().post_process(outputs, meta)
        if self.class_conf_thresholds:
            outputs = [
                [self._filter_class_confidence(tile[0]), *tile[1:]]
                for tile in outputs
            ]
        result = super().post_process(outputs, meta)
        if self.copper_max_aspect_ratio is None:
            return result
        keep = []
        for index, (box, name) in enumerate(zip(result.boxes, result.class_names)):
            if name == "brass_plate_6":
                x1, y1, x2, y2 = box
                if (x2 - x1) / (y2 - y1) > self.copper_max_aspect_ratio:
                    continue
            keep.append(index)
        return DetectResult(
            boxes=[result.boxes[index] for index in keep],
            scores=[result.scores[index] for index in keep],
            class_ids=[result.class_ids[index] for index in keep],
            class_names=[result.class_names[index] for index in keep],
            ori_img=result.ori_img,
        )
