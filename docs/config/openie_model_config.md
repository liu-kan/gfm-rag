## LLM OpenIE Model Configuration

An example of a LLM OpenIE model configuration file is shown below:

```yaml
_target_: gfmrag.kg_construction.openie_model.LLMOPENIEModel
llm_api: openai
model_name: gpt-4o-mini
max_ner_tokens: 300
max_triples_tokens: 4096
```

|      Parameter       |                       Options                       |                                           Note                                           |
| :------------------: | :-------------------------------------------------: | :--------------------------------------------------------------------------------------: |
|      `_target_`      |                        None                         | The class name of [LLM OpenIE model][gfmrag.kg_construction.openie_model.LLMOPENIEModel] |
|      `llm_api`       | ``openai``, ``together``, ``ollama``, ``llama.cpp`` |                            The API to use for the LLM model.                             |
|     `model_name`     |                        None                         |               The name of the LLM model to use. For example, `gpt-4o-mini`               |
|   `max_ner_tokens`   |                        None                         |                  The maximum number of tokens to use for the NER model.                  |
| `max_triples_tokens` |                        None                         |                The maximum number of tokens to use for the triples model.                |

Please refer to [LLMOPENIEModel][gfmrag.kg_construction.openie_model.LLMOPENIEModel] for more details of the other parameters.
