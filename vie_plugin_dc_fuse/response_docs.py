"""场景响应文档：合成示例，保留实际业务字段和判定语义。"""

from schemas.data_base import DetectionItem, MoMResult


RESPONSE_NOTES = (
    '默认五路有熔丝盒有磁环使用v6.2切图模型，横梁明细为crossbeam_screw和'
    'no_crossbeam_screw；横梁合计必须恰好4个且无缺件标签，不单独保证上下各2个。'
    '其他型号保留原模型和上下横梁类别规则。'
    '### 场景明细与判定规则\n\n明细 scene 使用检测标签（screw_1、no_screw_1、nut_2、no_nut2、small_screw_'
    '8、no_small_screw_8、brass_plate_6、metal_piece_4、upper_crossbeam_screw_9、no_up'
    'per_crossbeam_screw_9、lower_crossbeam_screw_10、no_lower_crossbeam_screw_10）；'
    '同一判定项的所有检测框共用该项结论，数量不满足时即使检测框置信度很高仍为 FAIL。整体依据产品型号启用的数量和缺陷规则判定，不是对 detailLis'
    't 简单求和：缺失目标可能没有对应明细，空检测结果仍为 FAIL。name 当前为空字符串，accuracy 为目标检测置信度。示例对应六路无熔丝盒无磁'
    '环：12 个螺母、6 个铜排、4 个铁片；不通过示例缺少一个螺母。result.message 当前可能仍为“检测成功”，它表示流程完成，不能代替 ve'
    'rdict。\n\n本场景默认 coordinate 为原图宽高归一化的四边形八个数：[x1,y1,x2,y2,x3,y3,x4,y4]，矩形按左上、右上、'
    '右下、左下排列；x 乘原图宽、y 乘原图高可还原像素。缺失项可以为 []，不得当作原点检测框。color 是显示颜色，不作为判定依据；以 verdict'
    ' 为准。vis_image 为可选 JPEG data URI，关闭可视化或绘制失败时可为空；示例使用空字符串，不填伪造 base64。 当前这些示例场'
    '景的公共 ocr_tokens 为 null；不要把内部 OCR 结构当作既有接口字段。'
)


def _result(passed):
    items = []
    for label, count in [("nut_2", 12 if passed else 11), ("metal_piece_4", 4), ("brass_plate_6", 6)]:
        for index in range(count):
            x = 0.02 + index * 0.07
            items.append(DetectionItem(
                status=passed if label == "nut_2" else True, scene=label,
                coordinate=[x, 0.2, x + 0.04, 0.2, x + 0.04, 0.3, x, 0.3],
                accuracy=0.96,
            ))
    for item in items:
        item.coordinate = [round(value, 4) for value in item.coordinate]
    return MoMResult(status=passed, message="检测成功", detailList=items).to_dict()


RESPONSE_EXAMPLES = {
    "PASS": {"summary": "通过：六路型号的数量规则满足", "result": _result(True)},
    "FAIL": {"summary": "不通过：螺母数量不足，高置信度不等于合格", "result": _result(False)},
    "EMPTY_FAIL": {"summary": "未检出目标：空明细仍不通过", "result": MoMResult(status=False, message="检测成功").to_dict()},
}
