import logging
import math
import os
from itertools import islice

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F  # noqa:N812
from torch.utils import data as torch_data
from tqdm import tqdm

from deep_graphrag import utils
from deep_graphrag.models import UltraQA
from deep_graphrag.ultra import query_utils
from deep_graphrag.ultra.variadic import variadic_softmax

# A logger for this file
logger = logging.getLogger(__name__)

separator = ">" * 30
line = "-" * 30


def train_and_validate(
    cfg: DictConfig,
    output_dir: str,
    model: nn.Module,
    train_datasets: dict,
    valid_datasets: dict,
    device: torch.device,
    batch_per_epoch: int | None = None,
) -> None:
    if cfg.train.num_epoch == 0:
        return

    world_size = utils.get_world_size()
    rank = utils.get_rank()

    # Create dataloader for each dataset
    train_dataloader_dict = {}
    for data_name, dataset in train_datasets.items():
        train_data = dataset["data"]
        sampler = torch_data.DistributedSampler(train_data, world_size, rank)
        train_loader = torch_data.DataLoader(
            train_data, cfg.train.batch_size, sampler=sampler
        )
        train_dataloader_dict[data_name] = train_loader
    data_name_list = list(train_dataloader_dict.keys())

    batch_per_epoch = batch_per_epoch or len(train_loader)

    optimizer = get_class(cfg.optimizer["_target_"])(
        model.parameters(),
        **{k: v for k, v in cfg.optimizer.items() if k != "_target_"},
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if utils.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []

            for dataloader in train_dataloader_dict.values():
                dataloader.sampler.set_epoch(epoch)
            # np.random.seed(epoch)  # TODO: should we use the same dataloader for all processes?
            shuffled_data_name_list = np.random.permutation(
                data_name_list
            )  # Shuffle the dataloaders
            for data_name in shuffled_data_name_list:
                train_loader = train_dataloader_dict[data_name]
                graph = train_datasets[data_name]["graph"]
                ent2docs = train_datasets[data_name]["ent2docs"]
                entities_weight = None
                if cfg.model.init_entities_weight:
                    entities_weight = utils.get_entities_weight(ent2docs)
                for batch in tqdm(
                    islice(train_loader, batch_per_epoch),
                    desc=f"Training Batches: {epoch}",
                    total=batch_per_epoch,
                ):
                    batch = query_utils.cuda(batch, device=device)
                    pred = parallel_model(graph, batch, entities_weight=entities_weight)
                    target = batch[
                        "supporting_entities_masks"
                    ]  # supporting_entities_mask
                    loss = F.binary_cross_entropy_with_logits(
                        pred, target, reduction="none"
                    )
                    is_positive = target > 0.5
                    is_negative = target <= 0.5
                    num_positive = is_positive.sum(dim=-1)
                    num_negative = is_negative.sum(dim=-1)

                    neg_weight = torch.zeros_like(pred)
                    neg_weight[is_positive] = (
                        1 / num_positive.float()
                    ).repeat_interleave(num_positive)

                    if cfg.task.adversarial_temperature > 0:
                        with torch.no_grad():
                            logit = pred[is_negative] / cfg.task.adversarial_temperature
                            neg_weight[is_negative] = variadic_softmax(
                                logit, num_negative
                            )
                            # neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                    else:
                        neg_weight[is_negative] = (
                            1 / num_negative.float()
                        ).repeat_interleave(num_negative)
                    loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                    loss = loss.mean()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    if utils.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                        logger.warning(separator)
                        logger.warning(f"binary cross entropy: {loss:g}")
                    losses.append(loss.item())
                    batch_id += 1

            if utils.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning(f"average binary cross entropy: {avg_loss:g}")

        epoch = min(cfg.train.num_epoch, i + step)
        utils.synchronize()

        if cfg.train.do_eval:
            if rank == 0:
                logger.warning(separator)
                logger.warning("Evaluate on valid")
            result = test(cfg, model, valid_datasets, device=device)
        else:
            result = float("inf")
            best_result = float("-inf")
        if rank == 0:
            if result > best_result:
                best_result = result
                best_epoch = epoch
                logger.warning("Save checkpoint to model_best.pth")
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, os.path.join(output_dir, "model_best.pth"))
            if not cfg.train.save_best_only:
                logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(
                    state, os.path.join(output_dir, "model_epoch_%d.pth" % epoch)
                )
            logger.warning(f"Best mrr: {best_result:g} at epoch {best_epoch}")

    if rank == 0:
        logger.warning("Load checkpoint from model_best.pth")
    state = torch.load(
        os.path.join(output_dir, "model_best.pth"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state["model"])
    utils.synchronize()


@torch.no_grad()
def test(
    cfg: DictConfig,
    model: nn.Module,
    test_datasets: dict,
    device: torch.device,
    return_metrics: bool = False,
) -> float | dict:
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    # process sequentially of test datasets
    all_metrics = {}
    all_mrr = []
    for data_name, q_data in test_datasets.items():
        test_data = q_data["data"]
        graph = q_data["graph"]
        ent2docs = q_data["ent2docs"]
        sampler = torch_data.DistributedSampler(test_data, world_size, rank)
        test_loader = torch_data.DataLoader(
            test_data, cfg.train.batch_size, sampler=sampler
        )

        model.eval()
        ent_preds = []
        ent_targets = []
        doc_preds = []
        doc_targets = []

        # Create doc retriever
        doc_ranker_args = {
            key: value for key, value in cfg.doc_ranker.items() if key != "_target_"
        }
        doc_ranker = get_class(cfg.doc_ranker._target_)(
            ent2doc=ent2docs, **doc_ranker_args
        )

        entities_weight = None
        if cfg.model.init_entities_weight:
            entities_weight = utils.get_entities_weight(ent2docs)

        for batch in tqdm(test_loader):
            batch = query_utils.cuda(batch, device=device)
            ent_pred = model(graph, batch, entities_weight=entities_weight)
            doc_pred = doc_ranker(ent_pred)  # Ent2docs mapping
            target_entities_mask = batch[
                "supporting_entities_masks"
            ]  # supporting_entities_mask
            target_docs_mask = batch["supporting_docs_masks"]  # supporting_docs_mask
            target_entities = target_entities_mask.bool()
            target_docs = target_docs_mask.bool()
            ent_ranking, target_ent_ranking = utils.batch_evaluate(
                ent_pred, target_entities
            )
            doc_ranking, target_doc_ranking = utils.batch_evaluate(
                doc_pred, target_docs
            )

            # answer set cardinality prediction
            ent_prob = F.sigmoid(ent_pred)
            num_pred = (ent_prob * (ent_prob > 0.5)).sum(dim=-1)
            num_target = target_entities_mask.sum(dim=-1)
            ent_preds.append((ent_ranking, num_pred))
            ent_targets.append((target_ent_ranking, num_target))

            # document set cardinality prediction
            doc_prob = F.sigmoid(doc_pred)
            num_pred = (doc_prob * (doc_prob > 0.5)).sum(dim=-1)
            num_target = target_docs_mask.sum(dim=-1)
            doc_preds.append((doc_ranking, num_pred))
            doc_targets.append((target_doc_ranking, num_target))

        ent_pred = query_utils.cat(ent_preds)
        ent_target = query_utils.cat(ent_targets)
        doc_pred = query_utils.cat(doc_preds)
        doc_target = query_utils.cat(doc_targets)

        ent_pred, ent_target = utils.gather_results(
            ent_pred, ent_target, rank, world_size, device
        )
        doc_pred, doc_target = utils.gather_results(
            doc_pred, doc_target, rank, world_size, device
        )

        metrics = {}
        if rank == 0:
            ent_metrics = utils.evaluate(ent_pred, ent_target, cfg.task.metric)
            doc_metrics = utils.evaluate(doc_pred, doc_target, cfg.task.metric)
            for key, value in ent_metrics.items():
                metrics[f"ent_{key}"] = value
            for key, value in doc_metrics.items():
                metrics[f"doc_{key}"] = value
            metrics["mrr"] = (1 / ent_pred[0].float()).mean().item()
            logger.warning(f"{'-' * 15} Test on {data_name} {'-' * 15}")
            query_utils.print_metrics(metrics, logger)
        else:
            metrics["mrr"] = (1 / ent_pred[0].float()).mean().item()
        all_metrics[data_name] = metrics
        all_mrr.append(metrics["mrr"])
    utils.synchronize()
    all_avg_mrr = np.mean(all_mrr)
    return all_avg_mrr if not return_metrics else metrics


@hydra.main(config_path="config", config_name="stage2_qa_finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    utils.init_distributed_mode()
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")

    qa_datasets = utils.get_multi_dataset(cfg)
    device = utils.get_device()

    rel_emb_dim = {qa_data.rel_emb_dim for qa_data in qa_datasets.values()}
    assert (
        len(rel_emb_dim) == 1
    ), "All datasets should have the same relation embedding dimension"

    model = UltraQA(cfg.model.entity_model_cfg, rel_emb_dim=rel_emb_dim.pop())

    train_datasets = {}
    valid_datasets = {}

    for data_name, qa_data in qa_datasets.items():
        train_data, valid_data = qa_data._data
        graph = qa_data.kg.to(device)
        ent2docs = qa_data.ent2docs.to(device)
        train_datasets[data_name] = {
            "data": train_data,
            "graph": graph,
            "ent2docs": ent2docs,
        }
        valid_datasets[data_name] = {
            "data": valid_data,
            "graph": graph,
            "ent2docs": ent2docs,
        }

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)

    model = model.to(device)
    train_and_validate(
        cfg,
        output_dir,
        model,
        train_datasets,
        valid_datasets,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
    )

    if cfg.train.do_eval:
        if utils.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        test(cfg, model, valid_datasets, device=device)

    # Save the model into the format for QA inference
    if (
        utils.is_main_process()
        and cfg.train.save_pretrained
        and cfg.train.num_epoch > 0
    ):
        pre_trained_dir = os.path.join(output_dir, "pretrained")
        utils.save_model_to_pretrained(model, cfg, pre_trained_dir)

    if utils.get_world_size() > 1:
        utils.synchronize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
