import json
import pickle
import re

import hydra
import numpy as np
import pandas as pd
from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig
from omegaconf import DictConfig
from scipy.sparse import csr_array
from tqdm import tqdm


def processing_phrases(phrase: str) -> str:
    return re.sub("[^A-Za-z0-9 ]", " ", phrase.lower()).strip()


def get_searcher(
    phrases: np.ndarray,
    dataset: str,
    exp_name: str = "phrase",
    index_name: str = "nbits_2",
) -> Searcher:
    checkpoint_path = "checkpoints/exp/colbertv2.0"

    colbert_config = {
        "root": f"data/{dataset}/tmp/lm_vectors/colbert",
        "doc_index_name": "nbits_2",
        "phrase_index_name": "nbits_2",
    }

    phrases = [processing_phrases(p) for p in phrases]
    with Run().context(
        RunConfig(nranks=1, experiment=exp_name, root=colbert_config["root"])
    ):
        config = ColBERTConfig(
            nbits=2,
            root=colbert_config["root"],
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=phrases, overwrite="reuse")

    with Run().context(
        RunConfig(nranks=1, experiment=exp_name, root=colbert_config["root"])
    ):
        config = ColBERTConfig(
            root=colbert_config["root"],
        )
        phrase_searcher = Searcher(
            index=colbert_config["phrase_index_name"], config=config, verbose=1
        )

    return phrase_searcher


def link_node_by_colbertv2(
    queries_named_entities: pd.DataFrame, phrases: np.ndarray, dataset: str
) -> dict:
    """
    linking the ner entities from query to KGs by colbertv2. ==> question entities.
    """
    if "query" in queries_named_entities:
        queries_named_entities = {
            row["query"]: eval(row["triples"])
            for i, row in queries_named_entities.iterrows()
        }
    elif "question" in queries_named_entities:
        queries_named_entities = {
            row["question"]: eval(row["triples"])
            for i, row in queries_named_entities.iterrows()
        }

    phrase_searcher = get_searcher(phrases, dataset)

    question_entities = {}
    for question, entities in tqdm(
        queries_named_entities.items(), total=len(queries_named_entities.items())
    ):
        query_ner_list = entities["named_entities"]
        query_ner_list = [processing_phrases(p) for p in query_ner_list]

        phrase_ids = []
        for query in query_ner_list:
            queries = Queries(path=None, data={0: query})
            ranking = phrase_searcher.search_all(queries, k=1)

            for phrase_id, _rank, _score in ranking.data[0]:
                phrase_ids.append(phrase_id)

        question_entity = [phrases[phrase_id] for phrase_id in phrase_ids]
        question_entities[question] = question_entity

    return question_entities


def mapping_doc_to_phrases(
    dataset: str,
    doc_to_phrases_mat: csr_array,
    corpus: dict,
    phrases: list,
) -> dict:
    rows, cols = np.nonzero(doc_to_phrases_mat.toarray())
    unique_rows = list(range(doc_to_phrases_mat.toarray().shape[0]))

    if dataset in ["hotpotqa", "2wikimultihopqa", "musique"]:
        items = list(corpus.keys())
        doc2entities = {
            items[row]: phrases[cols[rows == row].tolist()].tolist()
            for row in unique_rows
        }

    return doc2entities


def generate_qa_dataset(
    data: dict,
    doc2entities: dict,
    question_entities: dict,
    dataset: str,
    mode: str,
) -> None:
    qa_dataset = []
    for sample in tqdm(data, total=len(data)):
        id = sample["_id"] if "_id" in sample else sample["id"]
        answer = sample["answer"]
        question = sample["question"]
        if dataset in ["hotpotqa", "2wikimultihopqa", "musique"]:
            supporting_facts = sample["supporting_facts"]

            supporting_entities = []
            for item in list(set(supporting_facts)):
                supporting_entities.extend(doc2entities[item])

        qa_dataset.append(
            {
                "id": id,
                "question": question,
                "answer": answer,
                "supporting_facts": supporting_facts,
                "question_entities": question_entities[question],
                "supporting_entities": supporting_entities,
            }
        )
    dataset_qa_path = f"data/{dataset}/processed/stage1/{mode}.json"
    json.dump(qa_dataset, open(dataset_qa_path, "w"), indent="\t")


@hydra.main(
    config_path="config", config_name="stage1_kg_construction", version_base=None
)
def main(cfg: DictConfig) -> None:
    dataset = cfg.dataset.data_name
    extraction_type = cfg.task.openie_cfg.type

    graph_type = cfg.task.create_graph.graph_type  # 'facts_and_sim'
    phrase_type = cfg.task.create_graph.phrase_type  # 'ents_only_lower_preprocess'
    version = cfg.task.create_graph.version

    corpus = json.load(open(f"data/{dataset}/raw/dataset_corpus.json"))
    train_data = json.load(open(f"data/{dataset}/raw/train.json"))
    test_data = json.load(open(f"data/{dataset}/raw/test.json"))

    # all the entities
    kb_node_phrase_to_id = pickle.load(
        open(
            f"data/{dataset}/tmp/{dataset}_{graph_type}_graph_phrase_dict_{phrase_type}_{extraction_type}.{version}.subset.p",
            "rb",
        )
    )
    phrases = np.array(list(kb_node_phrase_to_id.keys()))[
        np.argsort(list(kb_node_phrase_to_id.values()))
    ]

    # query ner results
    queries_named_entities = pd.read_csv(
        f"data/{dataset}/tmp/{dataset}_queries.named_entity_output.tsv", sep="\t"
    )
    question_entities = link_node_by_colbertv2(queries_named_entities, phrases, dataset)
    print("Finish extracting question entities...")

    # document2entities.json, {'document': entities}
    docs_to_facts_mat = pickle.load(
        open(
            f"data/{dataset}/tmp/{dataset}_{graph_type}_graph_doc_to_facts_csr_{phrase_type}_{extraction_type}.{version}.subset.p",
            "rb",
        )
    )  # (num docs, num facts)
    facts_to_phrases_mat = pickle.load(
        open(
            f"data/{dataset}/tmp/{dataset}_{graph_type}_graph_facts_to_phrases_csr_{phrase_type}_{extraction_type}.{version}.subset.p",
            "rb",
        )
    )  # (num facts, num phrases)
    doc_to_phrases_mat = docs_to_facts_mat.dot(facts_to_phrases_mat)
    doc_to_phrases_mat[doc_to_phrases_mat.nonzero()] = 1  # entities2documents

    doc2entities = mapping_doc_to_phrases(dataset, doc_to_phrases_mat, corpus, phrases)
    json.dump(
        doc2entities,
        open(f"data/{dataset}/processed/stage1/document2entities.json", "w"),
        indent="\t",
    )

    generate_qa_dataset(
        train_data, doc2entities, question_entities, dataset, mode="train"
    )
    generate_qa_dataset(
        test_data, doc2entities, question_entities, dataset, mode="test"
    )


if __name__ == "__main__":
    main()
