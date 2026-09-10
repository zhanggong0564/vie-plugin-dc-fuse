from pydantic import BaseModel, Field, model_validator
from typing import List, Optional

from schemas.common import AICameraModel, VisualReferenceParams


class ModelParams(VisualReferenceParams):
    """modelParams 保留旧版产品型号以兼容现场请求。"""

    product_model: Optional[str] = Field(default=None, description="兼容旧版产品型号")


AICameraModels = AICameraModel


class DCFuseRequest(BaseModel):
    """请求中 json_data 对应的结构化模型。"""

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")
    AICameraModel: List[AICameraModels] = Field(
        default_factory=list,
        description="AICamera模型列表，不同版本可重复同一产品类型",
    )

    @model_validator(mode="after")
    def validate_product_type_parameter(self):
        product_type_models = [
            model
            for model in self.AICameraModel
            if model.AIParameterName == "产品类型"
        ]
        new_values = []
        for model in product_type_models:
            value = model.AIParameterValue
            if not isinstance(value, str) or not value.strip():
                raise ValueError("产品类型的 AIParameterValue 必须是非空字符串")
            new_values.append(value.strip())

        unique_new_values = set(new_values)
        old_value = (self.modelParams.product_model or "").strip() or None
        if not unique_new_values and old_value is None:
            raise ValueError("缺少产品类型参数")
        if len(unique_new_values) > 1:
            if old_value is None or old_value not in unique_new_values:
                raise ValueError("AICameraModel 包含多个不同的产品类型")
        elif old_value is not None and unique_new_values:
            if old_value not in unique_new_values:
                raise ValueError("新旧产品类型参数不一致")
        return self

    @property
    def product_type(self) -> str:
        old_value = (self.modelParams.product_model or "").strip()
        new_values = {
            model.AIParameterValue.strip()
            for model in self.AICameraModel
            if model.AIParameterName == "产品类型"
        }
        if old_value and (not new_values or len(new_values) > 1):
            return old_value
        return next(iter(new_values))
