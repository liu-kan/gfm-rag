
## Simple Ranker

```yaml
_target_: gfmrag.doc_rankers.SimpleRanker
```

## IDF Ranker

```yaml
_target_: gfmrag.doc_rankers.IDFWeightedRanker
```

## Top-k Ranker

```yaml
_target_: gfmrag.doc_rankers.TopKRanker
top_k: 10
```

## IDF Top-k Ranker

```yaml
_target_: gfmrag.doc_rankers.IDFWeightedTopKRanker
top_k: 20
```
