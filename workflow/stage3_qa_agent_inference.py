import json
import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from deep_graphrag import DeepGraphRAG
from deep_graphrag.llms import BaseLanguageModel
from deep_graphrag.prompt_builder import QAPromptBuilder
from deep_graphrag.ultra import query_utils

# A logger for this file
logger = logging.getLogger(__name__)


def agent_reasoning(
    cfg: DictConfig,
    graphrag_retriever: DeepGraphRAG,
    llm: BaseLanguageModel,
    qa_prompt_builder: QAPromptBuilder,
    query: str,
) -> dict:
    step = 1
    current_query = query
    thoughts: list[str] = []
    retrieved_docs = graphrag_retriever.retrieve(current_query, top_k=cfg.test.top_k)
    logs = []
    while step <= cfg.test.max_steps:
        message = qa_prompt_builder.build_input_prompt(
            current_query, retrieved_docs, thoughts
        )
        response = llm.generate_sentence(message)

        if isinstance(response, Exception):
            raise response from None

        thoughts.append(response)

        logs.append(
            {
                "step": step,
                "query": current_query,
                "retrieved_docs": retrieved_docs,
                "response": response,
                "thoughts": thoughts,
            }
        )

        if "So the answer is:" in response:
            break

        step += 1

        new_ret_docs = graphrag_retriever.retrieve(response, top_k=cfg.test.top_k)

        retrieved_docs_dict = {doc["title"]: doc for doc in retrieved_docs}
        for doc in new_ret_docs:
            if doc["title"] in retrieved_docs_dict:
                if doc["norm_score"] > retrieved_docs_dict[doc["title"]]["norm_score"]:
                    retrieved_docs_dict[doc["title"]]["score"] = doc["score"]
                    retrieved_docs_dict[doc["title"]]["norm_score"] = doc["norm_score"]
            else:
                retrieved_docs_dict[doc["title"]] = doc
        # Sort the retrieved docs by score
        retrieved_docs = sorted(
            retrieved_docs_dict.values(), key=lambda x: x["score"], reverse=True
        )
        # Only keep the top k
        retrieved_docs = retrieved_docs[: cfg.test.top_k]

    final_response = " ".join(thoughts)
    return {"response": final_response, "retrieved_docs": retrieved_docs, "logs": logs}


@hydra.main(
    config_path="config", config_name="stage3_qa_agent_inference", version_base=None
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    graphrag_retriever = DeepGraphRAG.from_config(cfg)
    llm = instantiate(cfg.llm)
    agent_prompt_builder = QAPromptBuilder(cfg.agent_prompt)
    qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)
    test_data = graphrag_retriever.qa_data.raw_test_data
    max_samples = 10
    with open(os.path.join(output_dir, "prediction.jsonl"), "w") as f:
        total = 0
        for sample in tqdm(test_data):
            query = sample["question"]
            result = agent_reasoning(
                cfg, graphrag_retriever, llm, agent_prompt_builder, query
            )
            retrieved_docs = result["retrieved_docs"]
            response = result["response"]
            if "So the answer is:" not in result["response"]:
                retrieved_docs = result["retrieved_docs"]
                message = qa_prompt_builder.build_input_prompt(query, retrieved_docs)
                response = llm.generate_sentence(message)

            result = {
                "id": sample["id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "answer_aliases": sample.get(
                    "answer_aliases", []
                ),  # Some datasets have answer aliases
                "response": response,
                "retrieved_docs": retrieved_docs,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()
            total += 1
            if total >= max_samples:
                break
    result_path = os.path.join(output_dir, "prediction.jsonl")
    # Evaluation
    evaluator = get_class(cfg.qa_evaluator["_target_"])(prediction_file=result_path)
    metrics = evaluator.evaluate()
    query_utils.print_metrics(metrics, logger)


if __name__ == "__main__":
    main()
