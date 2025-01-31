from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.nn import functional as F  # noqa:N812

from gfmrag.ultra.variadic import variadic_softmax


class BaseLoss(ABC):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        pass


class BCELoss(BaseLoss):
    def __init__(
        self, adversarial_temperature: float = 0, *args: Any, **kwargs: Any
    ) -> None:
        self.adversarial_temperature = adversarial_temperature

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        is_positive = target > 0.5
        is_negative = target <= 0.5
        num_positive = is_positive.sum(dim=-1)
        num_negative = is_negative.sum(dim=-1)

        neg_weight = torch.zeros_like(pred)
        neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(
            num_positive
        )

        if self.adversarial_temperature > 0:
            with torch.no_grad():
                logit = pred[is_negative] / self.adversarial_temperature
                neg_weight[is_negative] = variadic_softmax(logit, num_negative)
                # neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
        else:
            neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(
                num_negative
            )
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()
        return loss


class ListCELoss(BaseLoss):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        target_sum = target.sum(dim=-1)
        non_zero_target_mask = target_sum != 0  # Skip empty target
        target_sum = target_sum[non_zero_target_mask]
        pred = pred[non_zero_target_mask]
        target = target[non_zero_target_mask]
        pred_prob = torch.sigmoid(pred)  # B x N
        pred_prob_sum = pred_prob.sum(dim=-1, keepdim=True)  # B x 1
        loss = -torch.log((pred_prob / (pred_prob_sum + 1e-5)) + 1e-5) * target
        loss = loss.sum(dim=-1) / target_sum
        loss = loss.mean()
        return loss
