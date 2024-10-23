# mypy: ignore-errors

import torch

from deep_graphrag.ultra import variadic


def entities_to_mask(entities, num_nodes):
    mask = torch.zeros(num_nodes)
    mask[entities] = 1
    return mask


def evaluate(pred, target, metrics):
    ranking, num_pred = pred
    type, answer_ranking, num_easy, num_hard = target

    metric = {}
    for _metric in metrics:
        if _metric == "mrr":
            answer_score = 1 / ranking.float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
        elif _metric.startswith("recall@"):
            threshold = int(_metric[7:])
            answer_score = (ranking <= threshold).float()
            query_score = (
                variadic.variadic_sum(answer_score, num_hard) / num_hard.float()
            )
        elif _metric.startswith("hits@"):
            threshold = int(_metric[5:])
            answer_score = (ranking <= threshold).float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
        elif _metric == "mape":
            query_score = (num_pred - num_easy - num_hard).abs() / (
                num_easy + num_hard
            ).float()
        else:
            raise ValueError(f"Unknown metric `{_metric}`")

        score = query_score.mean()
        name = _metric
        metric[name] = score.item()

    return metric
