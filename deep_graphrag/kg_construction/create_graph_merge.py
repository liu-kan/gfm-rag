# mypy: ignore-errors
import argparse
import copy
import json
import os
import pickle
import re
from glob import glob

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from tqdm import tqdm

from .colbertv2_knn_merge import colbertv2_knn
from .processing import processing_phrases

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"


def create_graph_merge(
    dataset: str,
    extraction_type: str,
    extraction_model: str,
    retriever_name: str,
    processed_retriever_name: str,
    threshold: float = 0.9,
    create_graph_flag: bool = False,
    cosine_sim_edges: bool = True,
) -> None:
    version = "v3"
    inter_triple_weight = 1.0
    similarity_max = 1.0
    possible_files = glob(
        f"data/{dataset}/tmp/openie_{dataset}_results_{extraction_type}_{extraction_model}_*.json"
    )

    max_samples = np.max(
        [
            int(file.split(f"{extraction_model}_")[1].split(".json")[0])
            for file in possible_files
        ]
    )
    extracted_file = json.load(
        open(
            f"data/{dataset}/tmp/openie_{dataset}_results_{extraction_type}_{extraction_model}_{max_samples}.json"
        )
    )

    # train
    possible_files = glob(
        f"data/{dataset}2/tmp/openie_{dataset}2_results_{extraction_type}_{extraction_model}_*.json"
    )

    max_samples_train = np.max(
        [
            int(file.split(f"{extraction_model}_")[1].split(".json")[0])
            for file in possible_files
        ]
    )
    extracted_file_train = json.load(
        open(
            f"data/{dataset}2/tmp/openie_{dataset}2_results_{extraction_type}_{extraction_model}_{max_samples_train}.json"
        )
    )

    extracted_triples = (
        extracted_file["docs"] + extracted_file_train["docs"]
    )  # original + train (2000)
    if extraction_model != "gpt-3.5-turbo-1106":
        extraction_type = extraction_type + "_" + extraction_model
    phrase_type = (
        "ents_only_lower_preprocess"  # entities only, lower case, preprocessed
    )
    if cosine_sim_edges:
        graph_type = "facts_and_sim"  # extracted facts and similar phrases
    else:
        graph_type = "facts"

    passage_json = []
    phrases = []
    entities = []
    relations = {}
    incorrectly_formatted_triples = []
    triples_wo_ner_entity = []
    triple_tuples = []
    full_neighborhoods = {}
    correct_wiki_format = 0

    for _i, row in tqdm(enumerate(extracted_triples), total=len(extracted_triples)):
        row["passage"]
        row["extracted_entities"]
        ner_entities = [processing_phrases(p) for p in row["extracted_entities"]]

        triples = row["extracted_triples"]

        doc_json = row

        clean_triples = []
        unclean_triples = []
        doc_entities = set()

        # Populate Triples from OpenIE
        for triple in triples:
            if not isinstance(triple, list):
                continue
            triple = [str(s) for s in triple]

            if len(triple) > 1:
                if len(triple) != 3:
                    clean_triple = [processing_phrases(p) for p in triple]
                    incorrectly_formatted_triples.append(triple)
                    unclean_triples.append(triple)
                else:
                    clean_triple = [processing_phrases(p) for p in triple]
                    if "" in clean_triple:
                        print(clean_triple)
                        continue
                    clean_triples.append(clean_triple)
                    phrases.extend(clean_triple)

                    head_ent = clean_triple[0]
                    tail_ent = clean_triple[2]

                    if head_ent not in ner_entities and tail_ent not in ner_entities:
                        triples_wo_ner_entity.append(triple)

                    relations[(head_ent, tail_ent)] = clean_triple[1]

                    raw_head_ent = triple[0]
                    raw_tail_ent = triple[2]

                    entity_neighborhood = full_neighborhoods.get(raw_head_ent, set())
                    entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                    full_neighborhoods[raw_head_ent] = entity_neighborhood

                    entity_neighborhood = full_neighborhoods.get(raw_tail_ent, set())
                    entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                    full_neighborhoods[raw_tail_ent] = entity_neighborhood

                    for triple_entity in [clean_triple[0], clean_triple[2]]:
                        entities.append(triple_entity)
                        doc_entities.add(triple_entity)

        doc_json["entities"] = list(set(doc_entities))
        doc_json["clean_triples"] = clean_triples
        doc_json["noisy_triples"] = unclean_triples
        triple_tuples.append(clean_triples)

        passage_json.append(doc_json)

    print(f"Correct Wiki Format: {correct_wiki_format} out of {len(extracted_triples)}")

    # after processing the raw data sampled from the original dataset.
    queries_full = pd.read_csv(
        f"data/{dataset}/tmp/{dataset}_queries.named_entity_output.tsv", sep="\t"
    )
    queries_full_train = pd.read_csv(
        f"data/{dataset}2/tmp/{dataset}2_queries.named_entity_output.tsv", sep="\t"
    )
    queries_full = pd.concat([queries_full, queries_full_train], axis=0)
    if "hotpotqa" in dataset or "2wikimultihopqa" in dataset or "musique" in dataset:
        queries = json.load(open(f"data/{dataset}/raw/train.json"))  # dataset.
        queries_train = json.load(open(f"data/{dataset}2/raw/train.json"))
        queries = queries + queries_train
        questions = [q["question"] for q in queries]
        queries_full = (
            queries_full.set_index("question", drop=False)
            if "musique" in dataset or "2wikimultihopqa" in dataset
            else queries_full.set_index("0", drop=False)
        )

    queries_full = queries_full.loc[questions]

    q_entities = []
    q_entities_by_doc = []
    for doc_ents in tqdm(queries_full.triples):
        doc_ents = eval(doc_ents)["named_entities"]
        try:
            clean_doc_ents = [processing_phrases(p) for p in doc_ents]
        except Exception as e:
            print(e)
            clean_doc_ents = []
        q_entities.extend(clean_doc_ents)
        q_entities_by_doc.append(clean_doc_ents)

    unique_phrases = list(np.unique(entities))
    unique_relations = np.unique(list(relations.values()) + ["equivalent"])
    q_phrases = list(np.unique(q_entities))
    all_phrases = copy.deepcopy(unique_phrases)
    all_phrases.extend(q_phrases)  # all entities in corpus and query

    kb = pd.DataFrame(unique_phrases, columns=["strings"])
    kb2 = copy.deepcopy(kb)
    kb["type"] = "query"
    kb2["type"] = "kb"
    kb_full = pd.concat([kb, kb2])
    kb_full.to_csv(f"data/{dataset}/tmp_merge/kb_to_kb.tsv", sep="\t")

    rel_kb = pd.DataFrame(unique_relations, columns=["strings"])
    rel_kb2 = copy.deepcopy(rel_kb)
    rel_kb["type"] = "query"
    rel_kb2["type"] = "kb"
    rel_kb_full = pd.concat([rel_kb, rel_kb2])
    rel_kb_full.to_csv(f"data/{dataset}/tmp_merge/rel_kb_to_kb.tsv", sep="\t")

    query_df = pd.DataFrame(q_phrases, columns=["strings"])
    query_df["type"] = "query"
    kb["type"] = "kb"
    kb_query = pd.concat([kb, query_df])
    kb_query.to_csv(f"data/{dataset}/tmp_merge/query_to_kb.tsv", sep="\t")

    colbertv2_knn(dataset, "kb_to_kb.tsv")
    colbertv2_knn(dataset, "query_to_kb.tsv")

    if create_graph_flag:
        print("Creating Graph")

        node_json = [{"idx": i, "name": p} for i, p in enumerate(unique_phrases)]
        pd.DataFrame(unique_phrases)
        kb_phrase_dict = {p: i for i, p in enumerate(unique_phrases)}

        lose_facts = []

        for triples in triple_tuples:
            lose_facts.extend([tuple(t) for t in triples])

        lose_fact_dict = {f: i for i, f in enumerate(lose_facts)}
        fact_json = [
            {"idx": i, "head": t[0], "relation": t[1], "tail": t[2]}
            for i, t in enumerate(lose_facts)
        ]

        json.dump(
            lose_facts,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_documents_triplets.json",
                "w",
            ),
            indent="\t",
        )

        json.dump(
            passage_json,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_passage_chatgpt_openIE.{phrase_type}_{extraction_type}.{version}.subset.json",
                "w",
            ),
        )
        json.dump(
            node_json,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_nodes_chatgpt_openIE.{phrase_type}_{extraction_type}.{version}.subset.json",
                "w",
            ),
        )
        json.dump(
            fact_json,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_clean_facts_chatgpt_openIE.{phrase_type}_{extraction_type}.{version}.subset.json",
                "w",
            ),
        )

        pickle.dump(
            kb_phrase_dict,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_phrase_dict_{phrase_type}_{extraction_type}.{version}.subset.p",
                "wb",
            ),
        )
        pickle.dump(
            lose_fact_dict,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_fact_dict_{phrase_type}_{extraction_type}.{version}.subset.p",
                "wb",
            ),
        )

        graph_json = {}  # (Num Phrases, {Phrase: ('triples': num)})

        docs_to_facts = {}  # (Num Docs, Num Facts)
        facts_to_phrases = {}  # (Num Facts, Num Phrases)
        graph = {}  # (Num Phrases, Num Phrases)

        num_triple_edges = 0

        # Creating Adjacency and Document to Phrase Matrices
        for doc_id, triples in tqdm(enumerate(triple_tuples), total=len(triple_tuples)):
            doc_phrases = []
            fact_edges = []

            # Iterate over triples
            for triple in triples:
                triple = tuple(triple)
                fact_id = lose_fact_dict[triple]

                if len(triple) == 3:
                    triple[1]
                    triple = np.array(triple)[[0, 2]]
                    docs_to_facts[(doc_id, fact_id)] = 1  # documents to triples

                    for i, phrase in enumerate(triple):
                        phrase_id = kb_phrase_dict[phrase]
                        doc_phrases.append(phrase_id)

                        facts_to_phrases[(fact_id, phrase_id)] = 1  # triples to phrases

                        for phrase2 in triple[i + 1 :]:
                            phrase2_id = kb_phrase_dict[phrase2]

                            fact_edge_r = (phrase_id, phrase2_id)
                            fact_edge_l = (phrase2_id, phrase_id)

                            fact_edges.append(fact_edge_r)
                            fact_edges.append(fact_edge_l)

                            graph[fact_edge_r] = (
                                graph.get(fact_edge_r, 0.0) + inter_triple_weight
                            )  # cumsum weights
                            graph[fact_edge_l] = (
                                graph.get(fact_edge_l, 0.0) + inter_triple_weight
                            )

                            phrase_edges = graph_json.get(phrase, {})
                            edge = phrase_edges.get(phrase2, ("triple", 0))
                            phrase_edges[phrase2] = ("triple", edge[1] + 1)
                            graph_json[phrase] = phrase_edges

                            phrase_edges = graph_json.get(phrase2, {})
                            edge = phrase_edges.get(phrase, ("triple", 0))
                            phrase_edges[phrase] = ("triple", edge[1] + 1)
                            graph_json[phrase2] = phrase_edges

                            num_triple_edges += 1

        pickle.dump(
            docs_to_facts,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_doc_to_facts_{phrase_type}_{extraction_type}.{version}.subset.p",
                "wb",
            ),
        )
        pickle.dump(
            facts_to_phrases,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_facts_to_phrases_{phrase_type}_{extraction_type}.{version}.subset.p",
                "wb",
            ),
        )

        docs_to_facts_mat = csr_array(
            (
                [int(v) for v in docs_to_facts.values()],
                (
                    [int(e[0]) for e in docs_to_facts.keys()],
                    [int(e[1]) for e in docs_to_facts.keys()],
                ),
            ),
            shape=(len(triple_tuples), len(lose_facts)),
        )
        facts_to_phrases_mat = csr_array(
            (
                [int(v) for v in facts_to_phrases.values()],
                (
                    [e[0] for e in facts_to_phrases.keys()],
                    [e[1] for e in facts_to_phrases.keys()],
                ),
            ),
            shape=(len(lose_facts), len(unique_phrases)),
        )

        pickle.dump(
            docs_to_facts_mat,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_doc_to_facts_csr_{phrase_type}_{extraction_type}.{version}.subset.p",
                "wb",
            ),
        )
        pickle.dump(
            facts_to_phrases_mat,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_facts_to_phrases_csr_{phrase_type}_{extraction_type}.{version}.subset.p",
                "wb",
            ),
        )

        pickle.dump(
            graph,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_fact_doc_edges_{phrase_type}_{extraction_type}.{version}.subset.p",
                "wb",
            ),
        )

        print("Loading Vectors")

        # Expanding OpenIE triples with cosine similarity-based synonymy edges
        if cosine_sim_edges:
            if "colbert" in retriever_name:
                kb_similarity = pickle.load(
                    open(
                        f"data/{dataset}/tmp_merge/lm_vectors/colbert/nearest_neighbor_kb_to_kb.p",
                        "rb",
                    )
                )

            print("Augmenting Graph from Similarity")
            graph_plus = copy.deepcopy(graph)
            kb_similarity = {processing_phrases(k): v for k, v in kb_similarity.items()}

            synonym_candidates = []

            for phrase in tqdm(kb_similarity.keys(), total=len(kb_similarity)):
                synonyms = []
                if len(re.sub("[^A-Za-z0-9]", "", phrase)) > 2:
                    phrase_id = kb_phrase_dict.get(phrase, None)
                    if phrase_id is not None:
                        nns = kb_similarity[phrase]
                        num_nns = 0
                        for nn, score in zip(nns[0], nns[1]):
                            nn = processing_phrases(nn)
                            if score < threshold or num_nns > 100:
                                break
                            if nn != phrase:
                                phrase2_id = kb_phrase_dict.get(nn)
                                if phrase2_id is not None:
                                    phrase2 = nn

                                    sim_edge = (phrase_id, phrase2_id)
                                    synonyms.append((nn, score))

                                    relations[(phrase, phrase2)] = "equivalent"
                                    graph_plus[sim_edge] = similarity_max * score

                                    num_nns += 1

                                    phrase_edges = graph_json.get(phrase, {})
                                    edge = phrase_edges.get(phrase2, ("similarity", 0))
                                    if edge[0] == "similarity":
                                        phrase_edges[phrase2] = (
                                            "similarity",
                                            edge[1] + score,
                                        )
                                        graph_json[phrase] = phrase_edges

                synonym_candidates.append((phrase, synonyms))

            pickle.dump(
                synonym_candidates,
                open(
                    f"data/{dataset}/tmp_merge/{dataset}_similarity_edges_mean_{threshold}_thresh_{phrase_type}_{extraction_type}_{processed_retriever_name}.{version}.subset.p",
                    "wb",
                ),
            )
        else:
            graph_plus = graph

        pickle.dump(
            relations,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_relation_dict_{phrase_type}_{extraction_type}_{processed_retriever_name}.{version}.subset.p",
                "wb",
            ),
        )

        print("Saving Graph")

        synonymy_edges = {
            edge for edge in relations.keys() if relations[edge] == "equivalent"
        }

        stat_df = [
            ("Total Phrases", len(phrases)),
            ("Unique Phrases", len(unique_phrases)),
            ("Number of Individual Triples", len(lose_facts)),
            (
                "Number of Incorrectly Formatted Triples (ChatGPT Error)",
                len(incorrectly_formatted_triples),
            ),
            (
                "Number of Triples w/o NER Entities (ChatGPT Error)",
                len(triples_wo_ner_entity),
            ),
            ("Number of Unique Individual Triples", len(lose_fact_dict)),
            ("Number of Entities", len(entities)),
            ("Number of Relations", len(relations)),
            ("Number of Unique Entities", len(np.unique(entities))),
            ("Number of Synonymy Edges", len(synonymy_edges)),
            ("Number of Unique Relations", len(unique_relations)),
        ]

        print(pd.DataFrame(stat_df).set_index(0))

        if similarity_max == 1.0:
            pickle.dump(
                graph_plus,
                open(
                    f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_mean_{threshold}_thresh_{phrase_type}_{extraction_type}_{processed_retriever_name}.{version}.subset.p",
                    "wb",
                ),
            )
        else:
            pickle.dump(
                graph_plus,
                open(
                    f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_mean_{threshold}_thresh_{phrase_type}_{extraction_type}_sim_max_{similarity_max}_{processed_retriever_name}.{version}.subset.p",
                    "wb",
                ),
            )

        json.dump(
            graph_json,
            open(
                f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_chatgpt_openIE.{phrase_type}_{extraction_type}.{version}.subset.json",
                "w",
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--model_name", type=str, default="colbertv2")
    parser.add_argument("--extraction_model", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--create_graph", default=True)
    parser.add_argument("--extraction_type", type=str, default="ner")
    parser.add_argument("--cosine_sim_edges", default=True)

    args = parser.parse_args()
    dataset = args.dataset
    retriever_name = args.model_name
    processed_retriever_name = retriever_name.replace("/", "_").replace(".", "")
    extraction_model = args.extraction_model.replace("/", "_")
    threshold = args.threshold
    create_graph_flag = args.create_graph
    extraction_type = args.extraction_type
    cosine_sim_edges = args.cosine_sim_edges

    create_graph_merge(
        dataset,
        extraction_type,
        extraction_model,
        retriever_name,
        processed_retriever_name,
        threshold,
        create_graph_flag,
        cosine_sim_edges,
    )
