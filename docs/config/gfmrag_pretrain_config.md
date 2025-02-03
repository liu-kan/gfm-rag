# GFM-RAG Pre-training Configuration
An example configuration file for GFM pre-training is shown below:

```yaml
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
|        `__target__`        |  None   |                 [KGDataset][gfmrag.datasets.KGDataset]                  |
|        `cfgs.root`         |  None   |               root dictionary of the datasets saving path               |
|    `cfgs.force_rebuild`    |  None   |                  whether to force rebuild the dataset                   |
| `cfgs.text_emb_model_cfgs` |  None   | [text embedding model][text-embedding-model-configuration]configuration |
|       `train_names`        |  `[]`   |                     List of training dataset names                      |
|       `valid_names`        |  `[]`   |                    List of validation dataset names                     |

## GFM model configuration

|    Parameter     |            Options             |                          Note                          |
| :--------------: | :----------------------------: | :----------------------------------------------------: |
|   `__target__`   |              None              |        [QueryGNN][gfmrag.models.QueryGNN] model        |
|  `entity_model`  |              None              | [EntityNBFNet][gfmrag.ultra.models.EntityNBFNet] model |
|   `input_dim`    |              None              |              input dimension of the model              |
|  `hidden_dims`   |              `[]`              |             hidden dimensions of the model             |
|  `message_func`  |  `transe`,`rotate`,`distmult`  |             message function of the model              |
| `aggregate_func` | `pna`,`min`,`max`,`mean`,`sum` |            aggregate function of the model             |
|   `short_cut`    |        `True`, `False`         |                whether to use short cut                |
|   `layer_norm`   |        `True`, `False`         |               whether to use layer norm                |


## Loss configuration

|         Parameter         | Options |                     Note                      |
| :-----------------------: | :-----: | :-------------------------------------------: |
|      `num_negative`       |  None   |   number of negative samples for each query   |
|     `strict_negative`     |  None   |    whether to use strict negative sampling    |
| `adversarial_temperature` |  None   | adversarial temperature for negative sampling |
|         `metric`          |  `[]`   |       evaluation metrics for the model        |
|   `optimizer._target_`    |  None   |         torch optimizer for the model         |
|      `optimizer.lr`       |  None   |        learning rate for the optimizer        |


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
