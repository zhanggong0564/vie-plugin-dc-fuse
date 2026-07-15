# Changelog

本插件变更记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)
规范，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 变更

- 同步示例和配置中的模型路径，采用 `weights/{scene}/{task}_{arch}_v{N}` 命名规范。

## [0.1.0] - 2026-05-30

### 新增

- 将直流熔丝旧场景迁移为 `vie-plugin-dc-fuse` 独立插件。
- 新增插件配置、路由、业务逻辑、单元测试和运行示例。
- 新增框架与插件二进制 wheel 构建配置。
