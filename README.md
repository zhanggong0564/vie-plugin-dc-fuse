# vie-plugin-dc-fuse

直流熔丝装配检测插件。插件通过 `vie.plugins` entry point 注册到 VIE
`ScenarioRegistry`，检测模型由框架 runner factory 创建并随服务关闭释放。

## API

- 路径：`POST /api/v1/dcfuse_detect`
- 表单字段：`file` 为图片，`json_data` 为 JSON 字符串
- 关键参数：优先使用 `AICameraModel` 中 `AIParameterName`
  为“产品类型”的 `AIParameterValue`；兼容旧请求的
  `modelParams.product_model`
- 支持型号以 `business_logic.py` 的 `SUPPORTED_TYPES` 为准

`json_data` 示例：

```json
{
  "product": "直流熔丝",
  "type": "material-no",
  "modelParams": {
    "guide_line": [],
    "example_images": []
  },
  "AICameraModel": [{
    "Id": "registration-id",
    "Version": 1,
    "ModelFile": null,
    "AIParameterName": "产品类型",
    "AIParameterValue": "五路有熔丝盒无磁环"
  }]
}
```

## 模型与配置

| 配置 | 默认值 |
| --- | --- |
| 检测模型 | `./weights/dc_fuse/det_yolo_v6.onnx` |
| 置信度阈值 | `0.6` |

模型不随插件仓库提交。运行目录必须能解析上述相对路径。

## 安装与运行

在框架仓库根目录安装插件并运行示例：

```bash
conda run -n mobile_vision pip install -e plugins/vie-plugin-dc-fuse --no-deps
conda run -n mobile_vision python plugins/vie-plugin-dc-fuse/examples/run.py \
  /path/to/image.jpg 五路有熔丝盒无磁环
```

服务安装插件后会自动发现 entry point，无需在框架中增加场景映射。

## 测试

在本插件目录执行：

```bash
conda run -n mobile_vision env PYTHONPATH=../..:. python -m pytest tests/ -v
```

变更记录见 [CHANGELOG.md](CHANGELOG.md)。

## 默认四切图推理

切图检测器是原 YoloInfer 的扩展，统一位于框架 services/tiled_yolo.py；
切图元数据和相关辅助算法同文件维护，不在 services/base 中另设切图模块。

默认开启切图推理，使用 `./weights/dc_fuse/det_yolo_v6_split.onnx` 和分类置信度阈值。
无需设置环境变量即可使用；模型缺失时报告执行错误。
需要恢复整图推理时设置 `DC_FUSE_TILED_INFERENCE=false`，使用原 v6 模型及0.6阈值。
框架最低版本为2.2.8；权重文件独立于Git和插件wheel，通过服务权重目录交付。
切图权重SHA256：`af052674fb11b9daaf263218ac7217428d7ee3a6b6b103f1ef711503ca32ad54`。

在运行目录的 `.env` 或环境变量中设置：

```dotenv
DC_FUSE_TILED_INFERENCE=true
DC_FUSE_TILED_MODEL_PATH=./weights/dc_fuse/det_yolo_v6_split.onnx
DC_FUSE_TILED_CONF_THRESHOLD=0.4
DC_FUSE_TILE_OVERLAP=200
DC_FUSE_TILED_COPPER_MAX_ASPECT_RATIO=1.0
# 可单独覆盖某个类别；未覆盖字段保留下面列出的默认值。
DC_FUSE_TILED_CLASS_CONF_THRESHOLDS__BRASS_PLATE_6=0.75
```

这些字段使用 Pydantic Settings 校验：非法布尔值、阈值越界、负数重叠、
非整数重叠和空 split 模型路径会报错。整图模式仍使用原
`DC_FUSE_MODEL_PATH`、`DC_FUSE_CONF_THRESHOLD`；后者不控制切图阈值。

切图模式的正常件默认阈值如下：

| 类别 | 配置字段 | 默认阈值 |
| --- | --- | ---: |
| 大螺钉 | `screw_1` | 0.75 |
| 铜排 | `brass_plate_6` | 0.75 |
| 小螺钉 | `small_screw_8` | 0.70 |
| 金属件 | `metal_piece_4` | 0.45 |
| 上横梁螺钉 | `upper_crossbeam_screw_9` | 0.40 |
| 下横梁螺钉 | `lower_crossbeam_screw_10` | 0.40 |

`no_*` 缺件标签及 `nut_2` 使用 `DC_FUSE_TILED_CONF_THRESHOLD`，默认0.4。
该字段是未单独配置类别的回退阈值，不再作为所有类别的统一阈值。
分类配置支持 `DC_FUSE_TILED_CLASS_CONF_THRESHOLDS` JSON，或使用
`DC_FUSE_TILED_CLASS_CONF_THRESHOLDS__<类别名>` 单字段环境变量；
例如 `DC_FUSE_TILED_CLASS_CONF_THRESHOLDS__SCREW_1=0.76`。
未知类别字段、非数字、非有限数及超出[0,1]的阈值在加载阶段报错。

每块NMS前按预测的原始最高分类检查对应阈值；置信度必须严格大于阈值。
被拒绝的框不会改成第二高分的类别，也不会提前按大螺钉阈值过滤横梁螺钉。

处理顺序为：检测前处理进行 2×2 切图及各块 letterbox → batch=1 逐块推理 →
检测后处理进行分类置信度筛选、各块 NMS、坐标回映、同类去重及铜排宽高比过滤 → 原业务判定 → 原图坐标归一化。
横纵方向各 200 像素为相邻块的**总重叠宽度**；小图无法有效切分时使用单块。
单块 NMS 阈值为 0.5，任一块推理失败均按执行错误处理，不使用部分结果继续判定。

拼接后按“原框面积×检测置信度”排序，同分优先置信度、再优先面积。
同类别框只要有正面积交集，就保留排序较优的原始框；仅边缘或角点接触不去重。
已删除旧的拼接去重 IoU/覆盖率阈值配置；每块 NMS 的 IoU 阈值仍为 0.5。
只与已保留框比较，被删除的框不会继续链式抑制其他目标；
不依赖切缝位置、不合成外接框，不互删不同类别或正常件与 `no_*` 缺件标签。
真实相邻目标若同类检测框交叠，也会被去重，需结合真实样本验证少检风险。
该能力只支持普通框检测，不支持切图分割或旋转框。

拼接去重后，仅铜排类别使用原图像素坐标框的宽÷高限制：
大于 `DC_FUSE_TILED_COPPER_MAX_ASPECT_RATIO` 的框被过滤，默认上限1.0；
等于上限时保留，其他类别及缺件标签不受该规则影响。
该规则针对当前纵向铜排的拍摄方向；其他方向须单独验证并配置。

当前阈值依据171张实际OK、曾误判NG的样本选定；真实缺陷NG召回及其他型号
的真实模型效果尚未验证，不能仅依据OK样本的通过数宣称缺陷检测安全。
默认开启在新插件加载时生效；已有环境变量可覆盖默认值，远端服务需要单独发布。
