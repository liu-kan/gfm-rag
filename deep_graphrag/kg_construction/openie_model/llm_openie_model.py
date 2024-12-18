# Adapt from: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/openie_with_retrieval_option_parallel.py
import json
import logging
from itertools import chain
from typing import Literal

import numpy as np
from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_openai import ChatOpenAI

from deep_graphrag.kg_construction.langchain_util import init_langchain_model
from deep_graphrag.kg_construction.openie_extraction_instructions import (
    ner_prompts,
    openie_post_ner_prompts,
)
from deep_graphrag.kg_construction.utils import extract_json_dict

from .base_model import BaseOPENIEModel

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class LLMOPENIEModel(BaseOPENIEModel):
    def __init__(
        self,
        llm_api: Literal["openai", "together", "ollama", "llama.cpp"] = "openai",
        model_name: str = "gpt-4o-mini",
        max_ner_tokens: int = 1024,
        max_triples_tokens: int = 4096,
    ):
        self.llm_api = llm_api
        self.model_name = model_name
        self.max_ner_tokens = max_ner_tokens
        self.max_triples_tokens = max_triples_tokens

        self.client = init_langchain_model(llm_api, model_name)

    def ner(self, text: str) -> list:
        ner_messages = ner_prompts.format_prompt(user_input=text)

        try:
            if isinstance(self.client, ChatOpenAI):  # JSON mode
                chat_completion = self.client.invoke(
                    ner_messages.to_messages(),
                    temperature=0,
                    max_tokens=self.max_ner_tokens,
                    stop=["\n\n"],
                    response_format={"type": "json_object"},
                )
                response_content = chat_completion.content
                response_content = eval(response_content)

            elif isinstance(self.client, ChatOllama) or isinstance(
                self.client, ChatLlamaCpp
            ):
                response_content = self.client.invoke(ner_messages.to_messages())
                response_content = extract_json_dict(response_content)

            else:  # no JSON mode
                chat_completion = self.client.invoke(
                    ner_messages.to_messages(), temperature=0
                )
                response_content = chat_completion.content
                response_content = extract_json_dict(response_content)

            if "named_entities" not in response_content:
                response_content = []
            else:
                response_content = response_content["named_entities"]

        except Exception as e:
            logger.error(f"Error in extracting named entities: {e}")
            response_content = []

        return response_content

    def openie_post_ner_extract(self, text: str, entities: list) -> str:
        named_entity_json = {"named_entities": entities}
        openie_messages = openie_post_ner_prompts.format_prompt(
            passage=text, named_entity_json=json.dumps(named_entity_json)
        )
        try:
            if isinstance(self.client, ChatOpenAI):  # JSON mode
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
                response_content = self.client.invoke(openie_messages.to_messages())
                response_content = extract_json_dict(response_content)
                response_content = str(response_content)
            else:  # no JSON mode
                chat_completion = self.client.invoke(
                    openie_messages.to_messages(),
                    temperature=0,
                    max_tokens=self.max_triples_tokens,
                )
                response_content = chat_completion.content
                response_content = extract_json_dict(response_content)
                response_content = str(response_content)

        except Exception as e:
            logger.error(f"Error in OpenIE: {e}")
            response_content = "{}"

        return response_content

    def __call__(self, text: str) -> dict:
        """_summary_

        Args:
            text (str): _description_

        Returns:
            dict: _description_
        """
        res = {"passage": text, "extracted_entities": [], "extracted_triples": []}

        # ner_messages = ner_prompts.format_prompt(user_input=text)
        doc_entities = self.ner(text)
        try:
            doc_entities = list(np.unique(doc_entities))
        except Exception as e:
            logger.error(f"Results has nested lists: {e}")
            doc_entities = list(np.unique(list(chain.from_iterable(doc_entities))))

        triples = self.openie_post_ner_extract(text, doc_entities)
        res["extracted_entities"] = doc_entities
        try:
            res["extracted_triples"] = eval(triples)["triples"]
        except Exception:
            logger.error(f"Error in parsing triples: {triples}")

        return res
