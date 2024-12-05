import hydra
from omegaconf import DictConfig

from deep_graphrag.kg_construction.create_graph import create_graph
from deep_graphrag.kg_construction.create_graph_merge import create_graph_merge
from deep_graphrag.kg_construction.named_entity_extraction_parallel import (
    named_entity_extraction_parallel,
)
from deep_graphrag.kg_construction.openie_with_retrieval_option_parallel import (
    openie_parallel,
)


@hydra.main(
    config_path="config", config_name="stage1_kg_construction", version_base=None
)
def main(cfg: DictConfig) -> None:
    dataset = cfg.dataset.data_name
    model_name = cfg.task.openie_cfg.llm
    llm = cfg.task.openie_cfg.llm_api
    num_passages = cfg.task.openie_cfg.num_passages
    num_processes = cfg.num_processes
    run_ner = cfg.task.openie_cfg.run_ner

    extraction_type = cfg.task.openie_cfg.type
    retriever_name = cfg.task.create_graph.smodel
    processed_retriever_name = retriever_name.replace("/", "_").replace(".", "")
    extraction_model = model_name.replace("/", "_")
    threshold = cfg.task.openie_cfg.thresh
    create_graph_flag = cfg.task.create_graph.flag
    cosine_sim_edges = cfg.task.create_graph.cosine_sim_edges

    openie_parallel(model_name, llm, dataset, num_passages, num_processes, run_ner)
    named_entity_extraction_parallel(model_name, llm, dataset, num_processes)
    create_graph(
        dataset,
        extraction_type,
        extraction_model,
        retriever_name,
        processed_retriever_name,
        threshold,
        create_graph_flag,
        cosine_sim_edges,
    )
    create_graph_merge(
        dataset[:-1],
        extraction_type,
        extraction_model,
        retriever_name,
        processed_retriever_name,
        threshold,
        create_graph_flag,
        cosine_sim_edges,
    )


if __name__ == "__main__":
    main()
