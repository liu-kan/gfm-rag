from importlib import util
from pathlib import Path

_UTILS_PATH = Path(__file__).resolve().parents[1] / "gfmrag" / "kg_construction" / "utils.py"
_SPEC = util.spec_from_file_location("gfmrag.kg_construction.utils", _UTILS_PATH)
assert _SPEC and _SPEC.loader
_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

extract_json_dict = getattr(_MODULE, "extract_json_dict")
processing_phrases = getattr(_MODULE, "processing_phrases")


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
