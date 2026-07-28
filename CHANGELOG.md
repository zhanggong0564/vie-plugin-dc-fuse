# Changelog

本插件变更记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)
规范，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.1] - 2026-07-28

### 变更

- 简化检测项数量判定与结果组装，保持各型号启用项和判定规则不变。
- 忽略本地运行日志目录，避免生成文件污染插件工作区。
- 场景注册迁移到框架 `ScenarioRegistry`。
- YOLO 检测器改为接收框架创建的推理 runner，模型路径不再由模型对象加载。
- 初始化失败时关闭已创建 runner，服务和示例退出时释放模型资源。
- 新增 runner 注入、初始化回滚和关闭契约测试。
- 同步示例和配置中的模型路径，采用 `weights/{scene}/{task}_{arch}_v{N}` 命名规范。

## [0.1.0] - 2026-05-30

### 新增

- 将直流熔丝旧场景迁移为 `vie-plugin-dc-fuse` 独立插件。
- 新增插件配置、路由、业务逻辑、单元测试和运行示例。
- 新增框架与插件二进制 wheel 构建配置。
