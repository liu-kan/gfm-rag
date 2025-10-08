from importlib import import_module
from typing import Any

__all__ = ["GFMRetriever", "KGIndexer"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module 'gfmrag' has no attribute '{name}'")

    module_path = "gfmrag.gfmrag_retriever" if name == "GFMRetriever" else "gfmrag.kg_indexer"
    module = import_module(module_path)
    return getattr(module, name)
