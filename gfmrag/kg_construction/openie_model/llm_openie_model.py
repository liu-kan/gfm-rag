# Adapt from: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/openie_with_retrieval_option_parallel.py
import json
import logging
import time
from itertools import chain
from typing import Any, Literal, Optional

import numpy as np
from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_openai import ChatOpenAI

from gfmrag.kg_construction.langchain_util import init_langchain_model
from gfmrag.kg_construction.openie_extraction_instructions import (
    ner_prompts,
    openie_post_ner_prompts,
)
from gfmrag.kg_construction.utils import extract_json_dict

from .base_model import BaseOPENIEModel

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class LLMOPENIEModel(BaseOPENIEModel):
    """
    A class for performing Open Information Extraction (OpenIE) using Large Language Models.

    This class implements OpenIE functionality by performing Named Entity Recognition (NER)
    and relation extraction using various LLM backends like OpenAI, Together, Ollama, or llama.cpp.

    Args:
        llm_api (Literal["openai", "together", "ollama", "llama.cpp"]): The LLM backend to use.
            Defaults to "openai".
        model_name (str): Name of the specific model to use. Defaults to "gpt-4o-mini".
        base_url (Optional[str]): Base URL for the LLM API. Defaults to None.
        api_key (Optional[str]): API key for the LLM service. Defaults to None.
        max_ner_tokens (int): Maximum number of tokens for NER output. Defaults to 1024.
        max_triples_tokens (int): Maximum number of tokens for relation triples output.
            Defaults to 4096.

    Attributes:
        llm_api: The LLM backend being used
        model_name: Name of the model being used
        max_ner_tokens: Token limit for NER
        max_triples_tokens: Token limit for relation triples
        client: Initialized language model client

    Methods:
        ner: Performs Named Entity Recognition on input text
        openie_post_ner_extract: Extracts relation triples after NER
        __call__: Main method to perform complete OpenIE pipeline

    Examples:
        >>> openie_model = LLMOPENIEModel()
        >>> result = openie_model("Emmanuel Macron is the president of France")
        >>> print(result)
        {'passage': 'Emmanuel Macron is the president of France', 'extracted_entities': ['Emmanuel Macron', 'France'], 'extracted_triples': [['Emmanuel Macron', 'president of', 'France']]}
    """

    def __init__(
        self,
        llm_api: Literal[
            "openai", "nvidia", "together", "ollama", "llama.cpp"
        ] = "openai",
        model_name: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_ner_tokens: int = 1024,
        max_triples_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize LLM-based OpenIE model.

        Args:
            llm_api (Literal["openai", "nvidia", "together", "ollama", "llama.cpp"]): The LLM API provider to use.
                Defaults to "openai".
            model_name (str): Name of the language model to use. Defaults to "gpt-4o-mini".
            base_url (Optional[str]): Base URL for the LLM API. Defaults to None.
            api_key (Optional[str]): API key for the LLM service. Defaults to None.
            max_ner_tokens (int): Maximum number of tokens for NER processing. Defaults to 1024.
            max_triples_tokens (int): Maximum number of tokens for triple extraction. Defaults to 4096.

        Attributes:
            llm_api: The selected LLM API provider
            model_name: Name of the language model
            base_url: Base URL for the LLM API
            api_key: API key for the LLM service
            max_ner_tokens: Token limit for NER
            max_triples_tokens: Token limit for triples
            client: Initialized language model client
        """
        self.llm_api = llm_api
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_ner_tokens = _coerce_int(max_ner_tokens, 1024)
        self.max_triples_tokens = _coerce_int(max_triples_tokens, 4096)
        self.max_retries = max(1, _coerce_int(max_retries, 3))
        self.retry_delay = max(0.0, _coerce_float(retry_delay, 1.0))

        self.client = init_langchain_model(
            llm=llm_api,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key
        )

    @staticmethod
    def _truncate_text(raw: Any, limit: int = 300) -> str:
        if not isinstance(raw, str):
            raw = str(raw)
        raw = raw.replace("\n", "\\n")
        if len(raw) <= limit:
            return raw
        return raw[:limit] + "..."

    @staticmethod
    def _normalize_entity_list(entities: Any) -> list[str]:
        if isinstance(entities, list):
            cleaned = []
            for entity in entities:
                if entity is None:
                    continue
                text = str(entity).strip()
                if text:
                    cleaned.append(text)
            return cleaned
        if entities is None:
            return []
        text = str(entities).strip()
        return [text] if text else []

    def ner(self, text: str) -> list[str]:
        """
        Performs Named Entity Recognition (NER) on the input text using different LLM clients.

        Args:
            text (str): Input text to extract named entities from.

        Returns:
            list: A list of named entities extracted from the text. Returns empty list if extraction fails.

        Note:
            - For OpenAI client, uses JSON mode with specific parameters
            - For Ollama and LlamaCpp clients, extracts JSON from regular response
            - For other clients, extracts JSON from regular response without JSON mode
            - Handles exceptions by returning empty list and logging error
        """
        ner_messages = ner_prompts.format_prompt(user_input=text)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.llm_api == "openai" and isinstance(
                    self.client, ChatOpenAI
                ):  # JSON mode
                    chat_completion = self.client.invoke(
                        ner_messages.to_messages(),
                        temperature=0,
                        max_tokens=self.max_ner_tokens,
                        stop=["\n\n"],
                        response_format={"type": "json_object"},
                    )
                    response_content = chat_completion.content
                elif isinstance(self.client, ChatOllama) or isinstance(
                    self.client, ChatLlamaCpp
                ):
                    response_content = self.client.invoke(
                        ner_messages.to_messages()
                    ).content
                else:  # no JSON mode
                    chat_completion = self.client.invoke(
                        ner_messages.to_messages(), temperature=0
                    )
                    response_content = chat_completion.content

                parsed_response = extract_json_dict(response_content)
                if not parsed_response:
                    logger.warning(
                        "NER response missing JSON content (attempt %s/%s): %s",
                        attempt,
                        self.max_retries,
                        self._truncate_text(response_content),
                    )
                else:
                    entities = self._normalize_entity_list(
                        parsed_response.get("named_entities")
                    )
                    if entities:
                        return entities
                    logger.warning(
                        "NER response missing 'named_entities' key or empty list (attempt %s/%s): %s",
                        attempt,
                        self.max_retries,
                        self._truncate_text(parsed_response),
                    )
            except Exception as exc:
                last_error = exc
                logger.error(
                    "Error in extracting named entities (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        if last_error:
            logger.error(
                "Failed to extract named entities after %s attempts: %s",
                self.max_retries,
                last_error,
            )
        return []

    def openie_post_ner_extract(self, text: str, entities: list[str]) -> dict[str, Any]:
        """
        Extracts open information (triples) from text using LLM, considering pre-identified named entities.

        Args:
            text (str): The input text to extract information from.
            entities (list): List of pre-identified named entities in the text.

        Returns:
            str: JSON string containing the extracted triples. Returns empty JSON object "{}" if extraction fails.

        Raises:
            Exception: Logs any errors that occur during the extraction process.

        Notes:
            - For ChatOpenAI client, uses JSON mode for structured output
            - For ChatOllama and ChatLlamaCpp clients, extracts JSON from unstructured response
            - For other clients, extracts JSON from response content
            - Uses temperature=0 and configured max_tokens for consistent outputs
        """
        named_entity_json = {"named_entities": entities}
        openie_messages = openie_post_ner_prompts.format_prompt(
            passage=text, named_entity_json=json.dumps(named_entity_json)
        )
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.llm_api == "openai" and isinstance(
                    self.client, ChatOpenAI
                ):  # JSON mode
                    chat_completion = self.client.invoke(
                        openie_messages.to_messages(),
                        temperature=0,
                        max_tokens=self.max_triples_tokens,
                        response_format={"type": "json_object"},
                    )
                    response_content = chat_completion.content
                elif isinstance(self.client, ChatOllama) or isinstance(
                    self.client, ChatLlamaCpp
                ):
                    response_content = self.client.invoke(
                        openie_messages.to_messages()
                    ).content
                else:  # no JSON mode
                    chat_completion = self.client.invoke(
                        openie_messages.to_messages(),
                        temperature=0,
                        max_tokens=self.max_triples_tokens,
                    )
                    response_content = chat_completion.content

                parsed_response = extract_json_dict(response_content)
                if parsed_response and isinstance(parsed_response.get("triples"), list):
                    return parsed_response

                logger.warning(
                    "OpenIE response missing 'triples' list (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    self._truncate_text(
                        parsed_response if parsed_response else response_content
                    ),
                )
            except Exception as exc:
                last_error = exc
                logger.error(
                    "Error in OpenIE triple extraction (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        if last_error:
            logger.error(
                "Failed to extract triples after %s attempts: %s",
                self.max_retries,
                last_error,
            )
        return {"triples": []}

    def __call__(self, text: str) -> dict:
        """
        Perform OpenIE on the given text.

        Args:
            text (str): input text

        Returns:
            dict: dict of passage, extracted entities, extracted_triples

                - passage (str): input text
                - extracted_entities (list): list of extracted entities
                - extracted_triples (list): list of extracted triples
        """
        res = {"passage": text, "extracted_entities": [], "extracted_triples": []}

        # ner_messages = ner_prompts.format_prompt(user_input=text)
        doc_entities = self.ner(text)
        try:
            doc_entities = list(np.unique(doc_entities))
        except Exception as e:
            logger.error(f"Results has nested lists: {e}")
            doc_entities = list(np.unique(list(chain.from_iterable(doc_entities))))
        if not doc_entities:
            logger.warning(
                "No entities extracted. Possibly model not following instructions"
            )
        triples_result = self.openie_post_ner_extract(text, doc_entities)
        res["extracted_entities"] = doc_entities
        if isinstance(triples_result, dict):
            extracted_triples = triples_result.get("triples", [])
            if isinstance(extracted_triples, list):
                res["extracted_triples"] = extracted_triples
            else:
                logger.error(
                    "Triples payload is not a list; received type %s",
                    type(extracted_triples).__name__,
                )
        else:
            logger.error(
                "Unexpected triples result type: %s",
                type(triples_result).__name__,
            )

        return res
