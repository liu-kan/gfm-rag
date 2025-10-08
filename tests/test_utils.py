from gfmrag.kg_construction.utils import extract_json_dict, processing_phrases


def test_processing_phrases_preserves_non_ascii_characters() -> None:
    original = "苏铁类植物具有较厚的叶片。"
    processed = processing_phrases(original)
    assert "苏铁类植物具有较厚的叶片" in processed


def test_extract_json_dict_handles_python_like_payloads() -> None:
    payload = "Model output: {'named_entities': ['苏铁类植物', '木本被子植物']}"
    parsed = extract_json_dict(payload)
    assert parsed == {
        "named_entities": ["苏铁类植物", "木本被子植物"],
    }


def test_extract_json_dict_returns_none_when_missing_json() -> None:
    assert extract_json_dict("no json here") is None
