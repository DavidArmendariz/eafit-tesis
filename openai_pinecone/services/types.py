from typing import Literal

from pydantic import BaseModel


class QuestionResponse(BaseModel):
    question_id: str
    answer_string: list[str] | None
    answer_boolean: Literal["0", "1"] | None
    answer_date: str | None
    meta: str | None


class QuestionResponseStr(BaseModel):
    question_id: str
    answer_string: str
    meta: str | None
