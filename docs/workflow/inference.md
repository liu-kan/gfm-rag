GFM-RAG can be directly used for retrieval on a given dataset without fine-tuning. We provide an easy-to-use [GFMRetriever][gfmrag.GFMRetriever] interface for inference.

## Config
You need to create a configuration file for inference. Here is an [example](../../workflow/config/stage3_qa_ircot_inference.yaml):

??? example

    ```yaml title="workflow/config/stage3_qa_ircot_inference.yaml"
    hydra:
        run:
            dir: outputs/qa_agent_inference/${dataset.data_name}/${now:%Y-%m-%d}/${now:%H-%M-%S} # Output directory

    defaults:
        - _self_
        - doc_ranker: idf_topk_ranker # The document ranker to use
        - agent_prompt: hotpotqa_ircot # The agent prompt to use
        - qa_prompt: hotpotqa # The QA prompt to use
        - ner_model: llm_ner_model # The NER model to use
        - el_model: colbert_el_model # The EL model to use
        - qa_evaluator: hotpotqa # The QA evaluator to use

    seed: 1024

    dataset:
        root: ./data # data root directory
        data_name: hotpotqa_test # data name

    llm:
        _target_: gfmrag.llms.ChatGPT # The language model to use
        model_name_or_path: gpt-3.5-turbo # The model name or path
        retry: 5 # Number of retries

    graph_retriever:
        model_path: save_models/gfmrag_8M # Checkpoint path of the pre-trained GFM-RAG model
        doc_ranker: ${doc_ranker} # The document ranker to use
        ner_model: ${ner_model} # The NER model to usek
        el_model: ${el_model} # The EL model to use
        qa_evaluator: ${qa_evaluator} # The QA evaluator to use
        init_entities_weight: True # Whether to initialize the entities weight


    test:
        top_k: 10 # Number of documents to retrieve
        max_steps: 2 # Maximum number of steps
        max_test_samples: -1 # -1 for all samples
        resume: null # Resume from previous prediction
    ```

Details of the configuration parameters are explained in the [GFM-RAG Configuration][gfm-rag-configuration] page.

## Initialize GFMRetriever

```python
import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gfmrag import GFMRetriever

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="config", config_name="stage3_qa_ircot_inference", version_base=None
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    gfmrag_retriever = GFMRetriever.from_config(cfg)
```

## Document Retrieval

```python
docs = retriever.retrieve("Who is the president of France?", top_k=5)
```

## Question Answering

```python
from hydra.utils import instantiate
from gfmrag.llms import BaseLanguageModel
from gfmrag.prompt_builder import QAPromptBuilder

llm = instantiate(cfg.llm)
qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)

message = qa_prompt_builder.build_input_prompt(current_query, retrieved_docs)
answer = llm.generate_sentence(message)  # Answer: "Emmanuel Macron"
```

## GFM-RAG + Agent for Multi-step Retrieval
You can also integrate the GFM-RAG with arbitrary reasoning agents to perform multi-step RAG. Here is an example of [IRCOT](https://arxiv.org/abs/2212.10509) + GFM-RAG:

You can run the following command to perform multi-step reasoning:

[stage3_qa_ircot_inference.py](../../workflow/stage3_qa_ircot_inference.py)
```bash
python workflow/stage3_qa_ircot_inference.py
```

## Batch Retrieval
You can also perform batch retrieval with GFM-RAG with multi GPUs supports by running the following command:

[stage3_qa_inference.py](../../workflow/stage3_qa_inference.py)
```bash
python workflow/stage3_qa_inference.py
# Multi-GPU retrieval
torchrun --nproc_per_node=4 workflow/stage3_qa_inference.py
```
