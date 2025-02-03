# GFM-RAG Fine-tuning Configuration
An example configuration file for GFM fine-tuning is shown below:

```yaml
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

## General Configuration

| Parameter | Options |              Note               |
| :-------: | :-----: | :-----------------------------: |
| `run.dir` |  None   | The output directory of the log |

## Defaults

|    Parameter     | Options |                                 Note                                  |
| :--------------: | :-----: | :-------------------------------------------------------------------: |
| `text_emb_model` |  None   | The [text embedding model][text-embedding-model-configuration] to use |


## Training datasets

|         Parameter          | Options |                                  Note                                   |
| :------------------------: | :-----: | :---------------------------------------------------------------------: |
|        `__target__`        |  None   |                 [QADataset][gfmrag.datasets.QADataset]                  |
|        `cfgs.root`         |  None   |               root dictionary of the datasets saving path               |
|    `cfgs.force_rebuild`    |  None   |                  whether to force rebuild the dataset                   |
| `cfgs.text_emb_model_cfgs` |  None   | [text embedding model][text-embedding-model-configuration]configuration |
|       `train_names`        |  `[]`   |                     List of training dataset names                      |
|       `valid_names`        |  `[]`   |                    List of validation dataset names    |

## GFM model configuration

|    Parameter     |            Options             |                          Note                          |
| :--------------: | :----------------------------: | :----------------------------------------------------: |
|   `__target__`   |              None              |        [QueryGNN][gfmrag.models.QueryGNN] model        |
|  `entity_model`  |              None              | [EntityNBFNet][gfmrag.ultra.models.QueryNBFNet] model |
|   `input_dim`    |              None              |              input dimension of the model              |
|  `hidden_dims`   |              `[]`              |             hidden dimensions of the model             |
|  `message_func`  |  `transe`,`rotate`,`distmult`  |             message function of the model              |
| `aggregate_func` | `pna`,`min`,`max`,`mean`,`sum` |            aggregate function of the model             |
|   `short_cut`    |        `True`, `False`         |                whether to use short cut                |
|   `layer_norm`   |        `True`, `False`         |               whether to use layer norm                |


## Loss configuration

|         Parameter         | Options |                     Note                      |
| :-----------------------: | :-----: | :-------------------------------------------: |
```
|     `strict_negative`     |  None   |    whether to use strict negative sampling    |
| `metric` |  None   |    evaluation metrics to use    |
| `losses` |  None   |    list of losses to use    |
| `losses[].name` |  None   |    name of the loss    |
| `losses[]._target_` |  None   |    [loss function][gfmrag.losses.BaseLoss] to use    |
| `losses[].cfg` |  None   |    configuration of the loss    |
| `losses[].cfg.weight` |  None   |    weight of the loss    |
| `losses[].cfg.is_doc_loss` |  None   |    whether the loss is for document    |

## Optimizer configuration

|   Parameter   | Options |                        Note                        |
| :-----------: | :-----: | :------------------------------------------------: |
| `optimizer._target_` |  None   |         torch optimizer for the model         |
| `optimizer.lr` |  None   |        learning rate for the optimizer        |

## Training configuration

|     Parameter     | Options |                        Note                        |
| :---------------: | :-----: | :------------------------------------------------: |
|   `batch_size`    |  None   |            batch size for the training             |
|    `num_epoch`    |  None   |           number of epochs for training            |
|  `log_interval`   |  None   |         logging interval for the training          |
|    `fast_test`    |  None   |          number of samples for fast test           |
| `save_best_only`  |  None   | whether to save the best model based on the metric |
| `save_pretrained` |  None   |     whether to save the model for QA inference     |
| `batch_per_epoch` |  None   |      number of batches per epoch for training      |
|     `timeout`     |  None   |       timeout minutes for multi-gpu training       |
|   `checkpoint`    |  None   |          checkpoint path for the training          |
