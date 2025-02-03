# GFM-RAG Configuration
An example configuration file for GFM-RAG is shown below:

```yaml
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

## General Configuration

| Parameter | Options |              Note               |
| :-------: | :-----: | :-----------------------------: |
| `run.dir` |  None   | The output directory of the log |

## Defaults

|   Parameter    | Options |                                         Note                                          |
| :------------: | :-----: | :-----------------------------------------------------------------------------------: |
|  `ner_model`   |  None   |                  The config of the [ner_model](ner_model_config.md)                   |
|   `el_model`   |  None   |                   The config of the [el_model](el_model_config.md)                    |
|  `doc_ranker`  |  None   |                 The config of the [doc_ranker](doc_ranker_config.md)                  |
| `qa_evaluator` |  None   |                  The config of the [qa_evaluator][gfmrag.evaluation]                  |
| `agent_prompt` |  None   | The config of the [QAPromptBuilder][gfmrag.evaluation.QAPromptBuilder] used for IRCOT |
|  `qa_prompt`   |  None   |        The config of the [QAPromptBuilder][gfmrag.evaluation.QAPromptBuilder]         |


## Dataset

|  Parameter  | Options |          Note           |
| :---------: | :-----: | :---------------------: |
|   `root`    |  None   | The data root directory |
| `data_name` |  None   |      The data name      |


## LLM

|       Parameter       | Options |                           Note                           |
| :-------------------: | :-----: | :------------------------------------------------------: |
|      `_target_`       |  None   |         The [language model][gfmrag.llms] to use         |
| `model_name_or_path`  |  None   |                  The model name or path                  |
| Additional parameters |  None   | Parameters to initialize a [language model][gfmrag.llms] |

Please refer to the [LLMs][gfmrag.llms] page for more details.

## Graph Retriever

|       Parameter        |    Options     |                                Note                                |
| :--------------------: | :------------: | :----------------------------------------------------------------: |
|       `_target_`       |      None      |        The [graph retriever][gfmrag.graph_retriever] to use        |
|      `model_path`      |      None      |          Checkpoint path of the pre-trained GFM-RAG model          |
|      `doc_ranker`      |      None      |          The [document ranker][gfmrag.doc_rankers] to use          |
|      `ner_model`       |      None      |      The [NER model][gfmrag.kg_construction.ner_model] to use      |
|       `el_model`       |      None      | The [EL model][gfmrag.kg_construction.entity_linking_model] to use |
|     `qa_evaluator`     |      None      |                      The QA evaluator to use                       |
| `init_entities_weight` | `True`,`False` |             Whether to initialize the entities weight              |


## Test

|     Parameter      | Options |                          Note                          |
| :----------------: | :-----: | :----------------------------------------------------: |
|      `top_k`       |  None   |            Number of documents to retrieve             |
|    `max_steps`     |  None   |      Maximum number of steps, `1` for single step      |
| `max_test_samples` |  None   | Maximum number of samples to test (-1 for all samples) |
|      `resume`      |  None   |            Resume from previous prediction             |
