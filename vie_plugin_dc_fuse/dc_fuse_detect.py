"""直流熔丝检测器。"""

from services.inference import InferenceRunner
from services.yolo import YoloInfer


class DCFuseDetector(YoloInfer):
    def __init__(
        self,
        runner: InferenceRunner,
        confThreshold=0.5,
        nmsThreshold=0.5,
        task="det",
    ):
        super().__init__(
            nc=12,
            runner=runner,
            confThreshold=confThreshold,
            nmsThreshold=nmsThreshold,
            task=task,
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
