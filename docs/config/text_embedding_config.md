# Text Embedding Model Configuration

## Pre-train Text Embedding Model Configuration

This configuration supports most of the pre-train text embedding models of [SentenceTransformer](https://huggingface.co/sentence-transformers). An example of a DPR text embedding model configuration file is shown below:

```yaml
_target_: gfmrag.text_emb_models.BaseTextEmbModel
text_emb_model_name: BAAI/bge-large-en
normalize: True
batch_size: 32
query_instruct: "Represent this sentence for searching relevant passages: "
passage_instruct: null
model_kwargs: null
```

## Nvidia Embedding Model Configuration

This configuration supports the [Nvidia embedding models](https://huggingface.co/nvidia/NV-Embed-v2). An example of a Nvidia embedding model configuration file is shown below:

```yaml
_target_: gfmrag.text_emb_models.NVEmbedV2
text_emb_model_name: nvidia/NV-Embed-v2
normalize: True
batch_size: 32
query_instruct: "Instruct: Given a question, retrieve entities that can help answer the question\nQuery: "
passage_instruct: null
model_kwargs:
  torch_dtype: bfloat16
```
