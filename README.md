# Deep GraphRAG

## Dependencies

- Python 3.12
- CUDA 12.4

```bash
conda create -n deep_graphrag python=3.12
conda activate deep_graphrag
poetry install
conda install cuda-toolkit -c nvidia/label/cuda-12.4.1
```

## Installation

Install from pip
```bash
pip install deep-graphrag
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

[Optional] Install Llama.cpp
If you want to use Llama.cpp for locally held LLM, install it from the following repository.
https://github.com/abetlen/llama-cpp-python

### For development
Install pre-commit hooks
```bash
pre-commit install
# Example: pre-commit run --files path/to/file.py
```

## Workflow

### Stage1: Index Dataset
1. Create KG index for corpus.
2. Prepare QA dataset for training and evaluation (Optional)
```bash
python workflow/stage1_index_dataset.py
```


### Stage2: Deep GraphRAG Training

Unsupervised training on the constructed KG.

```bash
python workflow/stage2_kg_pretrain.py
# Multi-GPU training
torchrun --nproc_per_node=4 workflow/stage2_kg_pretrain.py
```

Supervised training on the QA dataset.

```bash
python workflow/stage2_qa_finetune.py
# Multi-GPU training
torchrun --nproc_per_node=4 workflow/stage2_qa_finetune.py
```

Evaluate retrieval performance of the trained model on QA dataset.

```bash
python workflow/stage2_qa_finetune.py train.num_epoch=0 datasets.train_names=[] checkpoint=save_models/qa_ultra_512_train_60000_w_pre-train/model.pth
# Multi-GPU evaluation
torchrun --nproc_per_node=4 workflow/stage2_qa_finetune.py train.num_epoch=0 datasets.train_names=[] checkpoint=save_models/qa_ultra_512_train_60000_w_pre-train/model.pth
```

### Stage3: QA Reasoning

#### Single Step QA Reasoning
```bash
python workflow/stage3_qa_inference.py
# Multi-GPU retrieval
torchrun --nproc_per_node=4 workflow/stage3_qa_inference.py
```

hotpotqa
```bash
torchrun --nproc_per_node=4 workflow/stage3_qa_inference.py dataset.data_name=hotpotqa_test qa_prompt=hotpotqa qa_evaluator=hotpotqa
```

musique
```bash
torchrun --nproc_per_node=4 workflow/stage3_qa_inference.py dataset.data_name=musique_test qa_prompt=musique qa_evaluator=musique
```

2Wikimultihopqa
```bash
torchrun --nproc_per_node=4 workflow/stage3_qa_inference.py dataset.data_name=2wikimultihopqa_test qa_prompt=2wikimultihopqa qa_evaluator=2wikimultihopqa
```
#### Multi Step IRCOT QA Reasoning
```bash
python workflow/stage3_qa_ircot_inference.py
```

hotpotqa
```bash
python workflow/stage3_qa_ircot_inference.py qa_prompt=hotpotqa qa_evaluator=hotpotqa agent_prompt=hotpotqa_ircot dataset.data_name=hotpotqa_test test.max_steps=2
```

musique
```bash
python workflow/stage3_qa_ircot_inference.py qa_prompt=musique qa_evaluator=musique agent_prompt=musique_ircot dataset.data_name=musique_test test.max_steps=4
```

2Wikimultihopqa
```bash
python workflow/stage3_qa_ircot_inference.py qa_prompt=2wikimultihopqa qa_evaluator=2wikimultihopqa agent_prompt=2wikimultihopqa_ircot dataset.data_name=2wikimultihopqa_test test.max_steps=2
```

### Visualize Paths
```bash
python workflow/experiments/visualize_path.py dataset.data_name=hotpotqa_test
```
