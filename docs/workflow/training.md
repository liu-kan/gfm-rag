# GFM-RAG Training

You can further fine-tune the pre-trained GFM-RAG model on your own dataset to improve the performance of the model on your specific domain.

## Data Preparation
Please follow the instructions in the [Data Preparation](data_preparation.md) to prepare your dataset in the following structure:
Make sure to have the `train.json` to perform the fine-tuning.

```
data_name/
├── raw/
│   ├── dataset_corpus.json
│   ├── train.json
│   └── test.json # (optional)
└── processed/
    └── stage1/
        ├── kg.txt
        ├── document2entities.json
        ├── train.json
        └── test.json # (optional)
```

## GFM Pre-training

During pre-training, the GFM model will sample triples from the KG-index `kg.txt` to construct synthetic queries and target entities for training.

A example of the KG-index:

```txt
fred gehrke,was,american football player
fred gehrke,was,executive
fred gehrke,played for,cleveland   los angeles rams
```

You need to create a configuration file for pre-training. Here is an [example](../../workflow/config/stage2_kg_pretrain.yaml):

??? example

    ```yaml title="workflow/config/stage2_kg_pretrain.yaml"
    hydra:
        run:
            dir: outputs/kg_pretrain/${now:%Y-%m-%d}/${now:%H-%M-%S} # Output directory

    defaults:
        - _self_
        - text_emb_model: mpnet # The text embedding model to use

    seed: 1024

    datasets:
        _target_: gfmrag.datasets.KGDataset # The KG dataset class
        cfgs:
            root: ./data # data root directory
            force_rebuild: False # Whether to force rebuild the dataset
            text_emb_model_cfgs: ${text_emb_model} # The text embedding model configuration
        train_names: # List of training dataset names
            - hotpotqa
        valid_names: []

    # GFM model configuration
    model:
        _target_: gfmrag.models.QueryGNN
        entity_model:
            _target_: gfmrag.ultra.models.EntityNBFNet
            input_dim: 512
            hidden_dims: [512, 512, 512, 512, 512, 512]
            message_func: distmult
            aggregate_func: sum
            short_cut: yes
            layer_norm: yes

    # Loss configuration
    task:
        num_negative: 256
        strict_negative: yes
        adversarial_temperature: 1
        metric: [mr, mrr, hits@1, hits@3, hits@10]

        optimizer:
        _target_: torch.optim.AdamW
        lr: 5.0e-4

    # Training configuration
    train:
        batch_size: 8
        num_epoch: 10
        log_interval: 100
        fast_test: 500
        save_best_only: no
        save_pretrained: no # Save the model for QA inference
        batch_per_epoch: null
        timeout: 60 # timeout minutes for multi-gpu training

    # Checkpoint configuration
    checkpoint: null
    ```

Details of the configuration parameters are explained in the [GFM-RAG Pre-training Config][gfm-rag-pre-training-configuration] page.

You can pre-train the GFM-RAG model on your dataset using the following command:

[stage2_kg_pretrain.py](../../workflow/stage2_kg_pretrain.py)
```bash
python workflow/stage2_kg_pretrain.py
# Multi-GPU training
torchrun --nproc_per_node=4 workflow/stage2_kg_pretrain.py
# Multi-node Multi-GPU training
torchrun --nproc_per_node=4 --nnodes=2 workflow/stage2_kg_pretrain.py
```

## GFM Fine-tuning

During fine-tuning, the GFM model will be trained on the query-documents pairs `train.json` from the labeled dataset to learn complex relationships for retrieval.

A example of the training data:

```json
[
	{
		"id": "5abc553a554299700f9d7871",
		"question": "Kyle Ezell is a professor at what School of Architecture building at Ohio State?",
		"answer": "Knowlton Hall",
		"supporting_facts": [
			"Knowlton Hall",
			"Kyle Ezell"
		],
		"question_entities": [
			"kyle ezell",
			"architectural association school of architecture",
			"ohio state"
		],
		"supporting_entities": [
			"10 million donation",
			"2004",
			"architecture",
			"austin e  knowlton",
			"austin e  knowlton school of architecture",
			"bachelor s in architectural engineering",
			"city and regional planning",
			"columbus  ohio  united states",
			"ives hall",
			"july 2002",
			"knowlton hall",
			"ksa",
		]
	},
    ...
]
```

You need to create a configuration file for fine-tuning. Here is an [example](../../workflow/config/stage2_qa_finetune.yaml):

??? example

    ```yaml title="workflow/config/stage2_qa_finetune.yaml"
    hydra:
        run:
            dir: outputs/qa_finetune/${now:%Y-%m-%d}/${now:%H-%M-%S} # Output directory

        defaults:
            - _self_
            - doc_ranker: idf_topk_ranker # The document ranker to use
            - text_emb_model: mpnet # The text embedding model to use

    seed: 1024

    datasets:
        _target_: gfmrag.datasets.QADataset # The QA dataset class
        cfgs:
            root: ./data # data root directory
            force_rebuild: False # Whether to force rebuild the dataset
            text_emb_model_cfgs: ${text_emb_model} # The text embedding model configuration
        train_names: # List of training dataset names
            - hotpotqa
        valid_names: # List of validation dataset names
            - hotpotqa_test
            - musique_test
            - 2wikimultihopqa_test

    # GFM model configuration
    model:
        _target_: gfmrag.models.GNNRetriever
        entity_model:
            _target_: gfmrag.ultra.models.QueryNBFNet
            input_dim: 512
            hidden_dims: [512, 512, 512, 512, 512, 512]
            message_func: distmult
            aggregate_func: sum
            short_cut: yes
            layer_norm: yes

    # Loss configuration
    task:
        strict_negative: yes
        metric: [mrr, hits@1, hits@2, hits@3, hits@5, hits@10, hits@20, hits@50, hits@100]
        losses:
            - name: ent_bce_loss
              loss:
                _target_: gfmrag.losses.BCELoss
                adversarial_temperature: 0.2
              cfg:
                weight: 0.3
                is_doc_loss: False
            - name: ent_pcr_loss
              loss:
                _target_: gfmrag.losses.ListCELoss
              cfg:
                weight: 0.7
                is_doc_loss: False


    # Optimizer configuration
    optimizer:
        _target_: torch.optim.AdamW
        lr: 5.0e-4

    # Training configuration
    train:
        batch_size: 8
        num_epoch: 20
        log_interval: 100
        batch_per_epoch: null
        save_best_only: yes
        save_pretrained: yes # Save the model for QA inference
        do_eval: yes
        timeout: 60 # timeout minutes for multi-gpu training
        init_entities_weight: True

        checkpoint: null
    ```

Details of the configuration parameters are explained in the [GFM-RAG Fine-tuning Configuration][gfm-rag-fine-tuning-configuration] page.


You can fine-tune the pre-trained GFM-RAG model on your dataset using the following command:

[stage2_qa_finetune.py](../../workflow/stage2_qa_finetune.py)
```bash
python workflow/stage2_qa_finetune.py
# Multi-GPU training
torchrun --nproc_per_node=4 workflow/stage2_qa_finetune.py
# Multi-node Multi-GPU training
torchrun --nproc_per_node=4 --nnodes=2 workflow/stage2_qa_finetune.py
```
