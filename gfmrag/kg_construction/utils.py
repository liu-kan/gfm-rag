import ast
import json
import os
import re
import unicodedata
from typing import Any

KG_DELIMITER = ","


def processing_phrases(phrase: Any) -> str:
    if phrase is None:
        return ""

    if isinstance(phrase, int):
        return str(phrase)  # deal with the int values

    if not isinstance(phrase, str):
        phrase = str(phrase)

    normalized = unicodedata.normalize("NFKC", phrase).casefold()
    cleaned_chars: list[str] = []

    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith(("L", "N")):
            cleaned_chars.append(char)
        elif category.startswith("M"):
            cleaned_chars.append(char)
        elif category.startswith("Z"):
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(" ")

    cleaned_text = "".join(cleaned_chars)
    return " ".join(cleaned_text.split())


def directory_exists(path: str) -> None:
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def extract_json_dict(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text

    if not isinstance(text, str) or not text.strip():
        return None

    pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}"
    match = re.search(pattern, text)

    if not match:
        return None

    json_string = match.group()
    parsers = (
        json.loads,
        ast.literal_eval,
    )
    for parser in parsers:
        try:
            parsed = parser(json_string)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            return parsed

    return None
