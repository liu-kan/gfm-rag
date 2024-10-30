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
python workflow/stage2_qa_finetune.py checkpoint=save_models/qa_ultra_epoch_20/model.pth train.num_epoch=0
# Multi-GPU evaluation
torchrun --nproc_per_node=4 workflow/stage2_qa_finetune.py checkpoint=save_models/qa_ultra_epoch_20/model.pth train.num_epoch=0
```
