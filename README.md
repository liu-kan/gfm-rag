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

Install faiss-gpu if you want to use Colbert for entity linking.
```bash
conda install -c pytorch -c nvidia faiss-gpu=1.9.0
```

### For development
Install pre-commit hooks
```bash
pre-commit install
# Example: pre-commit run --files path/to/file.py
```

## Workflow

### Stage1: KG Construction
Construct KG for corpus and query
```bash
python workflow/stage1_process_construction.py
```

Construct dataset for kgc task
```bash
python workflow/stage1_kg_construction.py
```

Construct dataset for qa reasoning task (Option)
```bash

python workflow/stage1_qa_construction.py
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
python workflow/stage2_qa_finetune.py train.num_epoch=0 datasets.train_names=[] checkpoint=save_models/qa_ultra_train_1000/model.pth
# Multi-GPU evaluation
torchrun --nproc_per_node=4 workflow/stage2_qa_finetune.py train.num_epoch=0 datasets.train_names=[] checkpoint=save_models/qa_ultra_train_1000/model.pth
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
