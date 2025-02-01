# KG-index Configuration

An example of a KG-index configuration file is shown below:

```yaml
hydra:
  run:
    dir: outputs/kg_construction/${now:%Y-%m-%d}/${now:%H-%M-%S} # Output directory

defaults:
  - _self_
  - ner_model: llm_ner_model # The NER model to use
  - openie_model: llm_openie_model # The OpenIE model to use
  - el_model: colbert_el_model # The EL model to use

dataset:
  root: ./data # data root directory
  data_name: hotpotqa # data name

kg_constructor:
  _target_: gfmrag.kg_construction.KGConstructor # The KGConstructor class
  open_ie_model: ${openie_model}
  ner_model: ${ner_model}
  el_model: ${el_model}
  root: tmp/kg_construction # Temporary directory for storing intermediate files during KG construction
  num_processes: 10 # Number of processes to use
  cosine_sim_edges: True # Whether to conduct entities resolution using cosine similarity
  threshold: 0.8 # Threshold for cosine similarity
  max_sim_neighbors: 100 # Maximum number of similar neighbors to add
  add_title: True # Whether to add the title to the content of the document during OpenIE
  force: False # Whether to force recompute the KG

qa_constructor:
  _target_: gfmrag.kg_construction.QAConstructor # The QAConstructor class
  root: tmp/qa_construction # Temporary directory for storing intermediate files during QA construction
  ner_model: ${ner_model}
  el_model: ${el_model}
  num_processes: 10 # Number of processes to use
  force: False # Whether to force recompute the QA data
```

## General Configuration

| Parameter | Options |              Note               |
| :-------: | :-----: | :-----------------------------: |
| `run.dir` |  None   | The output directory of the log |

## Defaults

|   Parameter    | Options |                            Note                             |
| :------------: | :-----: | :---------------------------------------------------------: |
|  `ner_model`   |  None   |    The config of the [ner_model](ner_model_config.md)    |
| `openie_model` |  None   | The config of the [openie_model](openie_model_config.md) |
|   `el_model`   |  None   |     The config of the [el_model](el_model_config.md)     |

## Dataset

|  Parameter  | Options |              Note               |
| :---------: | :-----: | :-----------------------------: |
|   `root`    |  None   | The data root directory |
| `data_name` |  None   | The data name |

## KG Constructor

|      Parameter      |     Options     |                                 Note                                  |
| :-----------------: | :-------------: | :-------------------------------------------------------------------: |
|     `_target_`      |      None       |          The class of [KGConstructor][gfmrag.kg_construction.KGConstructor]           |
|   `open_ie_model`   |      None       | The config of the [openie_model](openie_model_config.md) |
|     `ner_model`     |      None       |    The config of the [ner_model](ner_model_config.md)    |
|     `el_model`      |      None       |     The config of the [el_model](el_model_config.md)     |
|       `root`        |      None       | The temporary directory for storing intermediate files during KG construction |
|   `num_processes`   |      None       | The number of processes to use |
| `cosine_sim_edges`  |      None       | Whether to conduct entities resolution using cosine similarity |
|     `threshold`     |      None       | Threshold for cosine similarity |
| `max_sim_neighbors` |      None       | Maximum number of similar neighbors to add |
|     `add_title`     |      None       | Whether to add the title to the content of the document during OpenIE |
|      `force`        |      None       | Whether to force recompute the KG |


Please refer to [KG Constructor][gfmrag.kg_construction.KGConstructor] for details of parameters.


## QA Constructor

|      Parameter      |     Options     |                                 Note                                  |
| :-----------------: | :-------------: | :-------------------------------------------------------------------: |
|     `_target_`      |      None       |          The class of [QAConstructor][gfmrag.kg_construction.QAConstructor]           |
|       `root`        |      None       | The temporary directory for storing intermediate files during QA construction |
|     `ner_model`     |      None       |    The config of the [ner_model](ner_model_config.md)    |
|     `el_model`      |      None       |     The config of the [el_model](el_model_config.md)     |
|   `num_processes`   |      None       | The number of processes to use |
|      `force`        |      None       | Whether to force recompute the QA data |

Please refer to [QAConstructor][gfmrag.kg_construction.QAConstructor] for details of parameters.
