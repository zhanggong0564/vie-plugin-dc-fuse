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
