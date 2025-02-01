An example of a NER model configuration file is shown below:

```yaml
_target_: gfmrag.kg_construction.ner_model.LLMNERModel
llm_api: openai
model_name: gpt-4o-mini
max_tokens: 300
```

## LLM NER Model Configuration
|  Parameter   |                       Options                       |                                  Note                                   |
| :----------: | :-------------------------------------------------: | :---------------------------------------------------------------------: |
|  `_target_`  |                        None                         | The class name of [LLM NER model][gfmrag.kg_construction.ner_model.LLMNERModel] |
|  `llm_api`   | ``openai``, ``together``, ``ollama``, ``llama.cpp`` |                    The API to use for the LLM model.                    |
| `model_name` |                        None                         |      The name of the LLM model to use. For example, `gpt-4o-mini`       |
| `max_tokens` |                        None                         |         The maximum number of tokens to use for the LLM model.          |

Please refer to [LLM NER model][gfmrag.kg_construction.ner_model.LLMNERModel] for details on the other parameters.
