"""直流熔丝数据回流命名规则测试。"""

from pathlib import Path

import pytest

from vie_plugin_dc_fuse.plugin import dc_fuse_router


@pytest.mark.parametrize(
    "filename, expected_stem",
    [
        ("集中式-SG1100UD-AI拍照-直流侧-1-1-1789009471192.jpg", "1789009471192"),
        ("1789009471192.png", "1789009471192"),
        ("直流熔丝-末尾非数字.jpg", "直流熔丝-末尾非数字"),
        ("直流熔丝.jpg", "直流熔丝"),
    ],
)
def test_resolve_backflow_target_uses_trailing_numeric_timestamp(filename, expected_stem):
    target = dc_fuse_router.resolve_backflow_target(
        filename,
        fallback_product_type="六路无熔丝盒无磁环",
    )

    assert target.scene_dir == "dc_fuse"
    assert target.model_dir == "六路无熔丝盒无磁环"
    assert target.save_stem == expected_stem


@pytest.mark.parametrize("verdict_dir", ["ok", "ng", "pending"])
def test_backflow_paths_use_timestamp_for_image_and_record(tmp_path, verdict_dir):
    service = dc_fuse_router.backflow_service
    original_data_dir = service.data_dir
    service.data_dir = str(tmp_path)
    try:
        paths = service.resolve_paths(
            "直流熔丝-1789009471192.jpg",
            "2026-09-29T10:00:00.000",
            "六路无熔丝盒无磁环",
            verdict_dir,
            ".jpg",
        )
    finally:
        service.data_dir = original_data_dir

    expected_root = (
        tmp_path
        / "dc_fuse"
        / "2026-09-29"
        / "六路无熔丝盒无磁环"
        / verdict_dir
    )
    assert Path(paths["image_path"]) == expected_root / "images" / "1789009471192.jpg"
    assert Path(paths["record_path"]) == expected_root / "records" / "1789009471192.json"
