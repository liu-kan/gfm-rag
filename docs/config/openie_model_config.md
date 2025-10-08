## LLM OpenIE Model Configuration

An example of a LLM OpenIE model configuration file is shown below:

!!! example

    ```yaml title="gfmrag/workflow/config/openie_model/llm_openie_model.yaml"
    --8<-- "gfmrag/workflow/config/openie_model/llm_openie_model.yaml"
    ```


|      Parameter       |                             Options                             |                                           Note                                           |
| :------------------: | :-------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
|      `_target_`      |                              None                               | The class name of [LLM OpenIE model][gfmrag.kg_construction.openie_model.LLMOPENIEModel] |
|      `llm_api`       | ``openai``, ``nvidia``, ``together``, ``ollama``, ``llama.cpp`` |                            The API to use for the LLM model.                             |
|     `model_name`     |                              None                               |               The name of the LLM model to use. For example, `gpt-4o-mini`               |
|   `max_ner_tokens`   |                              None                               |                  The maximum number of tokens to use for the NER model.                  |
| `max_triples_tokens` |                              None                               |                The maximum number of tokens to use for the triples model.                |
|    `max_retries`     |                              None                               |  Number of retry attempts when an LLM request fails (also forwarded to the LLM client).  |
|    `retry_delay`     |                              None                               | The delay in seconds between retry attempts when the LLM call fails.                     |
|  `request_timeout`   |                              None                               | Timeout in seconds for each LLM request. Increase this when processing long passages.    |

Please refer to [LLMOPENIEModel][gfmrag.kg_construction.openie_model.LLMOPENIEModel] for more details of the other parameters.
