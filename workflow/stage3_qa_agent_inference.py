import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from deep_graphrag import DeepGraphRAG

# A logger for this file
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="config", config_name="stage3_qa_agent_inference", version_base=None
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    graphrag_agent = DeepGraphRAG.from_config(cfg)

    test_data = graphrag_agent.qa_data.raw_test_data
    for sample in test_data:
        query = sample["question"]
        graphrag_agent.retrieve(query, top_k=cfg.test.top_k)


if __name__ == "__main__":
    main()
