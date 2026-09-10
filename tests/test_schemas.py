import numpy as np
import pytest
from pydantic import ValidationError

from vie_plugin_dc_fuse.plugin import dc_fuse_router
from vie_plugin_dc_fuse.schemas import DCFuseRequest


def _request_payload(ai_camera_models, product_model="六路无熔丝盒无磁环"):
    return {
        "product": "直流熔丝",
        "type": "A0ST2147",
        "modelParams": {
            "product_model": product_model,
            "guide_line": [],
            "example_images": [],
        },
        "AICameraModel": ai_camera_models,
    }


def _camera_model(**overrides):
    model = {
        "Id": "registration-id",
        "Version": 1,
        "ModelFile": None,
        "AIParameterName": "产品类型",
        "AIParameterValue": "六路无熔丝盒无磁环",
    }
    model.update(overrides)
    return model


def test_product_type_comes_from_ai_camera_model():
    request = DCFuseRequest(**_request_payload([_camera_model()]))

    inputs = dc_fuse_router.get_inputs(request, np.zeros((2, 2, 3), np.uint8))

    assert request.product_type == "六路无熔丝盒无磁环"
    assert inputs.product_type == request.product_type
    assert dc_fuse_router._extract_product_type(request) == request.product_type


def test_product_type_falls_back_to_legacy_model_params():
    payload = _request_payload([], product_model="五路有熔丝盒有磁环")
    payload.pop("AICameraModel")

    request = DCFuseRequest(**payload)

    assert request.product_type == "五路有熔丝盒有磁环"


def test_matching_new_and_legacy_product_types_are_accepted():
    request = DCFuseRequest(**_request_payload([_camera_model()]))

    assert request.product_type == "六路无熔丝盒无磁环"


def test_repeated_matching_new_product_types_are_accepted():
    request = DCFuseRequest(
        **_request_payload(
            [_camera_model(), _camera_model(Id="another-registration-id")],
            product_model="",
        )
    )

    assert request.product_type == "六路无熔丝盒无磁环"


def test_legacy_product_type_disambiguates_multiple_new_values():
    request = DCFuseRequest(
        **_request_payload(
            [
                _camera_model(AIParameterValue="五路无熔丝盒无磁环"),
                _camera_model(Id="selected", AIParameterValue="六路无熔丝盒无磁环"),
            ],
            product_model="六路无熔丝盒无磁环",
        )
    )

    assert request.product_type == "六路无熔丝盒无磁环"


def test_conflicting_new_and_legacy_product_types_are_rejected():
    with pytest.raises(ValidationError, match="新旧产品类型参数不一致"):
        DCFuseRequest(
            **_request_payload(
                [_camera_model()],
                product_model="五路有熔丝盒有磁环",
            )
        )


@pytest.mark.parametrize(
    "ai_camera_models",
    [
        [_camera_model(AIParameterName="其他参数")],
        [_camera_model(AIParameterValue=None)],
        [_camera_model(AIParameterValue="  ")],
        [
            _camera_model(AIParameterValue="五路无熔丝盒无磁环"),
            _camera_model(Id="another-registration-id"),
        ],
    ],
)
def test_invalid_product_type_parameter_is_rejected(ai_camera_models):
    with pytest.raises(ValidationError):
        DCFuseRequest(**_request_payload(ai_camera_models, product_model=None))


def test_missing_product_type_is_rejected():
    with pytest.raises(ValidationError, match="缺少产品类型参数"):
        DCFuseRequest(**_request_payload([], product_model=None))
