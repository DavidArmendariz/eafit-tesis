import tiktoken


def num_tokens_from_string(string: str, encoding: tiktoken.Encoding) -> int:
    num_tokens = len(encoding.encode(string))
    return num_tokens
