from typing import Literal

question_ids = [2, 4, 25, 26, 29, 32, 37, 40]
question_ids_map = {
    "lessor_name": 2,
    "effective_date": 4,
    "commencement_date": 25,
    "end_date": 26,
    "option_to_purchase": 29,
    "option_to_extend": 32,
    "option_to_terminate_lessee": 37,
    "option_to_terminate_lessor": 40,
}
AvailableModels = Literal["gpt-3.5-turbo", "gpt-4o-2024-08-06", "gpt-4o-mini"]
AvailableEmbeddingModels = Literal[
    "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
]
