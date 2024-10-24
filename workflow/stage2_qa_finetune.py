import logging
import math
import os
from itertools import islice

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F  # noqa:N812
from torch.utils import data as torch_data
from torch_geometric.data import Data
from tqdm import tqdm

from deep_graphrag import utils
from deep_graphrag.datasets import QADataset
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
    graph: Data,
    train_data: torch.utils.data.Dataset,
    valid_data: torch.utils.data.Dataset,
    device: torch.device,
    batch_per_epoch: int | None = None,
) -> None:
    if cfg.train.num_epoch == 0:
        return

    world_size = utils.get_world_size()
    rank = utils.get_rank()

    sampler = torch_data.DistributedSampler(train_data, world_size, rank)
    train_loader = torch_data.DataLoader(
        train_data, cfg.train.batch_size, sampler=sampler
    )

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
            sampler.set_epoch(epoch)
            for batch in tqdm(
                islice(train_loader, batch_per_epoch),
                desc=f"Training Batches: {epoch}",
                total=batch_per_epoch,
            ):
                batch = query_utils.cuda(batch, device=device)
                pred = parallel_model(graph, batch)
                target = batch[-1]  # supporting_entities_mask
                loss = F.binary_cross_entropy_with_logits(
                    pred, target, reduction="none"
                )
                is_positive = target > 0.5
                is_negative = target <= 0.5
                num_positive = is_positive.sum(dim=-1)
                num_negative = is_negative.sum(dim=-1)

                neg_weight = torch.zeros_like(pred)
                neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(
                    num_positive
                )

                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        logit = pred[is_negative] / cfg.task.adversarial_temperature
                        neg_weight[is_negative] = variadic_softmax(logit, num_negative)
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
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(state, os.path.join(output_dir, "model_epoch_%d.pth" % epoch))
        utils.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, graph, valid_data, device=device)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load(
        os.path.join(output_dir, "model_epoch_%d.pth" % best_epoch), map_location=device
    )
    model.load_state_dict(state["model"])
    utils.synchronize()


@torch.no_grad()
def test(
    cfg: DictConfig,
    model: nn.Module,
    graph: Data,
    test_data: torch.utils.data.Dataset,
    device: torch.device,
    return_metrics: bool = False,
) -> float | dict:
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    sampler = torch_data.DistributedSampler(test_data, world_size, rank)
    test_loader = torch_data.DataLoader(
        test_data, cfg.train.batch_size, sampler=sampler
    )

    model.eval()
    preds = []
    targets = []
    for batch in tqdm(test_loader):
        batch = query_utils.cuda(batch, device=device)
        pred = model(graph, batch)
        target_mask = batch[-1]  # supporting_entities_mask
        type = torch.zeros(target_mask.shape[0]).long().to(pred)  # Just a placeholder
        target = (type, torch.zeros_like(target_mask).bool(), target_mask.bool())
        ranking, answer_ranking = query_utils.batch_evaluate(pred, target)
        # answer set cardinality prediction
        prob = F.sigmoid(pred)
        num_pred = (prob * (prob > 0.5)).sum(dim=-1)
        num_hard = target_mask.sum(dim=-1)
        num_easy = torch.zeros_like(num_hard)
        preds.append((ranking, num_pred))
        targets.append((type, answer_ranking, num_easy, num_hard))

    pred = query_utils.cat(preds)
    target = query_utils.cat(targets)

    pred, target = query_utils.gather_results(pred, target, rank, world_size, device)

    metrics = {}
    if rank == 0:
        metrics = utils.evaluate(pred, target, cfg.task.metric)
        query_utils.print_metrics(metrics, logger)
    else:
        metrics["mrr"] = (1 / pred[0].float()).mean().item()
    utils.synchronize()
    return metrics["mrr"] if not return_metrics else metrics


@hydra.main(config_path="config", config_name="stage2_qa_finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    utils.init_distributed_mode()
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")

    qa_data = QADataset(**cfg.dataset)
    device = utils.get_device()

    train_data, valid_data = qa_data._data
    graph = qa_data.kg
    rel_emb = graph.rel_emb
    graph = graph.to(device)

    model = UltraQA(cfg.model.entity_model_cfg, rel_emb_dim=rel_emb.shape[-1])

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    model = model.to(device)
    train_and_validate(
        cfg,
        output_dir,
        model,
        graph,
        train_data,
        valid_data,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
    )

    if utils.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, graph, valid_data, device=device)


if __name__ == "__main__":
    main()
