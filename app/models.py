from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @field_validator("*", mode="after")
    @classmethod
    def reject_blank_strings(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Строка не должна быть пустой.")
        return value


Scalar = str | int | float | bool | None


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    name: str = Field(max_length=200)
    manufacturer: str = Field(max_length=200)
    specs: dict[str, Scalar] = Field(max_length=50)
    description: str = Field(max_length=20000)

    @field_validator("specs")
    @classmethod
    def check_specs(cls, value: dict[str, Scalar]) -> dict[str, Scalar]:
        for key, item in value.items():
            if not key.strip() or len(key) > 120:
                raise ValueError("Название характеристики должно содержать от 1 до 120 символов.")
            if isinstance(item, str) and len(item) > 1000:
                raise ValueError("Значение характеристики длиннее 1000 символов.")
        return value


class Reference(StrictModel):
    field: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=1, max_length=20000)


class Claim(StrictModel):
    text: str = Field(min_length=1, max_length=2000)
    sources: list[Reference] = Field(min_length=1, max_length=8)


class MissingData(StrictModel):
    field: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=600)
    question: str = Field(min_length=1, max_length=400)
    blocking: bool


class Conflict(StrictModel):
    field: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=400)
    sources: list[Reference] = Field(min_length=2, max_length=4)


class ExtractedCharacteristic(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=1000)
    source: Reference


class Draft(StrictModel):
    status: Literal["ready", "needs_clarification"]
    seo_title: Claim | None
    seo_description: list[Claim] = Field(max_length=12)
    search_queries: list[Claim] = Field(max_length=5)
    audience: Claim | None
    tasks: Claim | None
    description_characteristics: list[ExtractedCharacteristic] = Field(max_length=20)
    missing_data: list[MissingData] = Field(max_length=30)
    conflicts: list[Conflict] = Field(max_length=20)


class Audit(StrictModel):
    supported: bool
    problems: list[str] = Field(max_length=30)
    missing_data: list[MissingData] = Field(max_length=30)
    conflicts: list[Conflict] = Field(max_length=20)


class Characteristic(StrictModel):
    name: str
    value: Scalar
    status: Literal["provided", "missing", "conflicting"]


class Commercial(StrictModel):
    audience: str
    tasks: str


class Lengths(StrictModel):
    seo_title: int
    seo_description: int


class ReadyResponse(StrictModel):
    status: Literal["ready"] = "ready"
    seo_title: str = Field(min_length=1, max_length=70)
    seo_description: str = Field(min_length=800, max_length=1200)
    key_characteristics: list[Characteristic]
    search_queries: list[str] = Field(min_length=3, max_length=5)
    commercial: Commercial
    missing_data: list[MissingData]
    conflicts: list[Conflict]
    lengths: Lengths


class ClarificationResponse(StrictModel):
    status: Literal["needs_clarification"] = "needs_clarification"
    seo_title: None = None
    seo_description: None = None
    key_characteristics: list[Characteristic]
    search_queries: list[str] = Field(max_length=0, default_factory=list)
    commercial: None = None
    missing_data: list[MissingData]
    conflicts: list[Conflict]
    lengths: None = None


GenerationResponse = Annotated[
    ReadyResponse | ClarificationResponse, Field(discriminator="status")
]
