This guide explains how to create a knowledge graph index and processing QA data that can be used by GFM-RAG.

## Data Preparation
Please follow the instructions in the [Data Preparation](data_preparation.md) to prepare your dataset in the following structure:

```
data_name/
├── raw/
│   ├── dataset_corpus.json
│   ├── train.json # (optional)
│   └── test.json # (optional)
└── processed/ # Output directory
```

## Config
You need to create a KG-index configuration file. Here is an example

!!! Example

    ```yaml
    hydra:
        run:
            dir: outputs/kg_construction/${now:%Y-%m-%d}/${now:%H-%M-%S} # Output directory

    defaults:
        - _self_
        - ner_model: llm_ner_model # The NER model to use
        - openie_model: llm_openie_model # The OpenIE model to use
        - el_model: colbert_el_model # The EL model to use

    dataset:
        root: ./data # data root directory
        data_name: hotpotqa # data name

    kg_constructor:
        _target_: gfmrag.kg_construction.KGConstructor # The KGConstructor class
        open_ie_model: ${openie_model}
        ner_model: ${ner_model}
        el_model: ${el_model}
        root: tmp/kg_construction # Temporary directory for storing intermediate files during KG construction
        num_processes: 10 # Number of processes to use
        cosine_sim_edges: True # Whether to conduct entities resolution using cosine similarity
        threshold: 0.8 # Threshold for cosine similarity
        max_sim_neighbors: 100 # Maximum number of similar neighbors to add
        add_title: True # Whether to add the title to the content of the document during OpenIE
        force: False # Whether to force recompute the KG

    qa_constructor:
        _target_: gfmrag.kg_construction.QAConstructor # The QAConstructor class
        root: tmp/qa_construction # Temporary directory for storing intermediate files during QA construction
        ner_model: ${ner_model}
        el_model: ${el_model}
        num_processes: 10 # Number of processes to use
        force: False # Whether to force recompute the QA data
    ```

Details of the configuration parameters are explained in the [KG-index Configuration][kg-index-configuration] page.

## Index dataset
To index the data, run the following command:

```bash
python workflow/stage1_index_dataset.py
```

You can overwrite the configuration like this:

```bash
python workflow/stage1_index_dataset.py +kg_constructor.num_processes=5
```

This method performs two main tasks:

1. Creates and saves knowledge graph related files (`kg.txt` and `document2entities.json`) from the `dataset_corpus.json` file
2. Identify the query entities and supporting entities in training and testing data if available in the raw data directory

Files created:

- `kg.txt`: Contains knowledge graph triples
- `document2entities.json`: Maps documents to their entities
- `train.json`: Processed training data (if raw exists)
- `test.json`: Processed test data (if raw exists)

Directory structure:
```
    root/
    └── data_name/
        ├── raw/
        │   ├── dataset_corpus.json
        │   ├── train.json (optional)
        │   └── test.json (optional)
        └── processed/
            └── stage1/
                ├── kg.txt
                ├── document2entities.json
                ├── train.json
                └── test.json
```

## Output Files

### `kg.txt`
The `kg.txt` file contains the knowledge graph triples in the following format:

```
subject,relation,object
```

Example:
```
fred gehrke,was,american football player
fred gehrke,was,executive
fred gehrke,played for,cleveland   los angeles rams
```

### `document2entities.json`
The `document2entities.json` file contains the mapping of documents to their entities in the following format:

- `key`: The title or unique id of the document.
- `value`: A list of entities in the document.

Example:
```json
{
    "Fred Gehrke": [
		"1977",
		"1981",
		"american football player",
		"chicago cardinals",
		"cleveland   los angeles rams",
		"executive",
		"first painted logo on helmets",
		"fred gehrke",
		"gehrke",
		"general manager of denver broncos",
		"los angeles rams",
		"los angeles rams logo",
		"miami marlin christian yelich",
		"san francisco 49ers"
	],
	"Manny Machado": [
		"2010 major league baseball draft",
		"american",
		"baltimore orioles",
		"brito high school",
		"july 6  1992",
		"major league baseball",
		"manny machado",
		"right handed"
	],
}
```

### `train.json` and `test.json`
The `train.json` and `test.json` files contain the processed training and test data in the following format:

- `id`: A unique identifier for the example.
- `question`: The question or query.
- `supporting_facts`: A list of supporting facts for the question. Each supporting fact is a list containing the title of the document that can be found in the `dataset_corpus.json` file.
- `question_entities`: A list of entities in the question.
- `supporting_entities`: A list of entities in the supporting facts.
- Additional fields copied from the raw data

Examples
```json
[
	{
		"id": "5abc553a554299700f9d7871",
		"question": "Kyle Ezell is a professor at what School of Architecture building at Ohio State?",
		"answer": "Knowlton Hall",
		"supporting_facts": [
			"Knowlton Hall",
			"Kyle Ezell"
		],
		"question_entities": [
			"kyle ezell",
			"architectural association school of architecture",
			"ohio state"
		],
		"supporting_entities": [
			"10 million donation",
			"2004",
			"architecture",
			"austin e  knowlton",
			"austin e  knowlton school of architecture",
			"bachelor s in architectural engineering",
			"city and regional planning",
			"columbus  ohio  united states",
			"ives hall",
			"july 2002",
			"knowlton hall",
			"ksa",
			"landscape architecture",
			"ohio",
			"replacement for ives hall",
			"the ohio state university",
			"the ohio state university in 1931",
			"american urban planning practitioner",
			"expressing local culture",
			"knowlton school",
			"kyle ezell",
			"lawrenceburg  tennessee",
			"professor",
			"the ohio state university",
			"theorist",
			"undergraduate planning program",
			"vibrant downtowns",
			"writer"
		]
	},
    ...
]
```
