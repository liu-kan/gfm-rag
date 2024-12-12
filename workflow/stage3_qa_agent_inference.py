import json
import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from deep_graphrag import DeepGraphRAG
from deep_graphrag.evaluation import RetrievalEvaluator
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
            retrieved_docs_dict.values(), key=lambda x: x["norm_score"], reverse=True
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
    max_samples = (
        cfg.test.max_test_samples if cfg.test.max_test_samples > 0 else len(test_data)
    )
    with open(os.path.join(output_dir, "prediction.jsonl"), "w") as f:
        for i in tqdm(range(max_samples)):
            sample = test_data[i]
            if i >= max_samples:
                break
            query = sample["question"]
            result = agent_reasoning(
                cfg, graphrag_retriever, llm, agent_prompt_builder, query
            )
            retrieved_docs = result["retrieved_docs"]

            # Generate QA response
            retrieved_docs = result["retrieved_docs"]
            message = qa_prompt_builder.build_input_prompt(query, retrieved_docs)
            qa_response = llm.generate_sentence(message)

            result = {
                "id": sample["id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "answer_aliases": sample.get(
                    "answer_aliases", []
                ),  # Some datasets have answer aliases
                "supporting_facts": sample["supporting_facts"],
                "response": qa_response,
                "retrieved_docs": retrieved_docs,
                "logs": result["logs"],
            }
            f.write(json.dumps(result) + "\n")
            f.flush()

    result_path = os.path.join(output_dir, "prediction.jsonl")
    # Evaluation
    evaluator = instantiate(cfg.qa_evaluator, prediction_file=result_path)
    metrics = evaluator.evaluate()
    query_utils.print_metrics(metrics, logger)

    # Eval retrieval results
    retrieval_evaluator = RetrievalEvaluator(prediction_file=result_path)
    retrieval_metrics = retrieval_evaluator.evaluate()
    query_utils.print_metrics(retrieval_metrics, logger)


if __name__ == "__main__":
    main()
