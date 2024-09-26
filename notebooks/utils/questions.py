import json
import re
from typing import List, TypedDict


class ProcessedQuestion(TypedDict):
    id: str
    page_reference: str
    display_question: str
    reader_question: str
    html_object_id: str | None
    restrictions: List[str] | None
    type: str
    response_mapping: dict | None
    user_response: str | None


def convert_text_to_json(text: str | None) -> dict | None:
    if text is None:
        return None
    else:
        # Use regex to replace Ruby-style hash syntax with JSON-style
        json_like_text = re.sub(r"'([^']+)'\s*=>\s*'([^']+)'", r'"\1": "\2"', text)
        return json.loads(json_like_text)


def json_to_dict(json_string) -> dict | None:
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None


def construct_single_question_for_ai(question) -> ProcessedQuestion:
    return {
        "id": question["id"],
        "page_reference": question["page_reference"],
        "display_question": question["display_question"],
        "reader_question": question["reader_question"],
        "html_object_id": question["html_object_id"],
        "restrictions": question["restrictions"],
        "type": question["type"],
        "response_mapping": convert_text_to_json(question["response_mapping"]),
        "user_response": question["user_response_source"],
    }


def construct_context(questions) -> List[ProcessedQuestion]:
    questions_array = []
    for question in questions:
        questions_array.append(construct_single_question_for_ai(question))
    return questions_array
