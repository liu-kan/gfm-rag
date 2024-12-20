import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from multiprocessing.dummy import Pool as ThreadPool
from typing import Any

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from deep_graphrag.kg_construction.utils import KG_DELIMITER, processing_phrases

from .entity_linking_model import BaseELModel
from .openie_model.base_model import BaseOPENIEModel

logger = logging.getLogger(__name__)


class BaseKGConstructor(ABC):
    @abstractmethod
    def create_kg(self, data_root: str, data_name: str) -> list[tuple[str, str, str]]:
        """
        Create a knowledge graph from the dataset

        Args:
            data_root (str): path to the dataset
            data_name (str): name of the dataset

        Returns:
            list[tuple[str, str, str]]: list of triples
        """
        pass

    @abstractmethod
    def get_document2entities(self, data_root: str, data_name: str) -> dict:
        """
        Get the document to entities mapping from the dataset

        Args:
            data_root (str): path to the dataset
            data_name (str): name of the dataset

        Returns:
            dict: document to entities mapping
        """
        pass


class KGConstructor(BaseKGConstructor):
    DELIMITER = KG_DELIMITER

    def __init__(
        self,
        open_ie_model: BaseOPENIEModel,
        el_model: BaseELModel,
        root: str = "tmp/kg_construnction",
        num_processes: int = 1,
        cosine_sim_edges: bool = True,
        threshold: float = 0.8,
        max_sim_neighbors: int = 100,
        add_title: bool = True,
        force: bool = False,
    ) -> None:
        self.open_ie_model = open_ie_model
        self.el_model = el_model
        self.root = root
        self.num_processes = num_processes
        self.cosine_sim_edges = cosine_sim_edges
        self.threshold = threshold
        self.max_sim_neighbors = max_sim_neighbors
        self.add_title = add_title
        self.force = force
        self.data_name = None

    @property
    def tmp_dir(self) -> str:
        assert (
            self.data_name is not None
        )  # data_name should be set before calling this property
        tmp_dir = os.path.join(self.root, self.data_name)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        return tmp_dir

    @staticmethod
    def from_config(cfg: DictConfig) -> "KGConstructor":
        # create a fingerprint of config for tmp directory
        config = OmegaConf.to_container(cfg, resolve=True)
        if "force" in config:
            del config["force"]
        fingerprint = hashlib.md5(json.dumps(config).encode()).hexdigest()

        base_tmp_dir = os.path.join(cfg.root, fingerprint)
        if not os.path.exists(base_tmp_dir):
            os.makedirs(base_tmp_dir)
            json.dump(
                config,
                open(os.path.join(base_tmp_dir, "config.json"), "w"),
                indent=4,
            )
        return KGConstructor(
            root=base_tmp_dir,
            open_ie_model=instantiate(cfg.open_ie_model),
            el_model=instantiate(cfg.el_model),
            num_processes=cfg.num_processes,
            cosine_sim_edges=cfg.cosine_sim_edges,
            threshold=cfg.threshold,
            max_sim_neighbors=cfg.max_sim_neighbors,
            add_title=cfg.add_title,
            force=cfg.force,
        )

    def create_kg(self, data_root: str, data_name: str) -> list[tuple[str, str, str]]:
        # Get dataset information
        self.data_name = data_name  # type: ignore
        raw_path = os.path.join(data_root, data_name, "raw")

        if self.force:
            # Clear cache in tmp directory
            for tmp_file in os.listdir(self.tmp_dir):
                os.remove(os.path.join(self.tmp_dir, tmp_file))

        open_ie_result_path = self.open_ie_extraction(raw_path)
        graph = self.create_graph(open_ie_result_path)
        extracted_triples = [(h, r, t) for (h, t), r in graph.items()]
        return extracted_triples

    def get_document2entities(self, data_root: str, data_name: str) -> dict:
        # Get dataset information
        self.data_name = data_name  # type: ignore

        if not os.path.exists(os.path.join(self.tmp_dir, "passage_info.json")):
            logger.warning(
                "Document to entities mapping is not available. Run create_kg first"
            )
            self.create_kg(data_root, data_name)

        with open(os.path.join(self.tmp_dir, "passage_info.json")) as fin:
            passage_info = json.load(fin)
        document2entities = {doc["title"]: doc["entities"] for doc in passage_info}
        return document2entities

    def open_ie_extraction(self, raw_path: str) -> str:
        """
        Extract open information extraction from the dataset corpus

        Args:
            raw_path (str): Path to the raw dataset

        Returns:
            str: Path to the openie results
        """
        # Read data corpus
        with open(os.path.join(raw_path, "dataset_corpus.json")) as f:
            corpus = json.load(f)
            if self.add_title:
                corpus = {
                    title: title + "\n" + passage for title, passage in corpus.items()
                }
        passage_to_title = {corpus[title]: title for title in corpus.keys()}

        logger.info(f"Numbers of passages: {len(corpus)}")

        open_ie_result_path = f"{self.tmp_dir}/openie_results.jsonl"
        open_ie_results = {}
        # check if the openie results are already computed
        if os.path.exists(open_ie_result_path):
            logger.info(f"OpenIE results already exists at {open_ie_result_path}")
            with open(open_ie_result_path) as f:
                for line in f:
                    data = json.loads(line)
                    open_ie_results[data["passage"]] = data

        remining_passages = [
            passage for passage in corpus.values() if passage not in open_ie_results
        ]
        logger.info(
            f"Numbers of passages which require processing: {len(remining_passages)}"
        )

        if len(remining_passages) > 0:
            with open(open_ie_result_path, "a") as f:
                with ThreadPool(processes=self.num_processes) as pool:
                    for result in tqdm(
                        pool.imap(self.open_ie_model, remining_passages),
                        total=len(remining_passages),
                        desc="Perform OpenIE",
                    ):
                        if isinstance(result, dict):
                            passage_title = passage_to_title[result["passage"]]
                            result["title"] = passage_title
                            f.write(json.dumps(result) + "\n")
                            f.flush()

        logger.info(f"OpenIE results saved to {open_ie_result_path}")
        return open_ie_result_path

    def create_graph(self, open_ie_result_path: str) -> dict:
        """
        Create a knowledge graph from the openie results

        Args:
            open_ie_result_path (str): Path to the openie results
        Returns:
            dict: Knowledge graph
                key: (head, tail)
                value: relation
        """

        with open(open_ie_result_path) as f:
            extracted_triples = [json.loads(line) for line in f]

        # Create a knowledge graph from the openie results
        passage_json = []  # document-level information
        phrases = []  # clean triples
        entities = []  # entities from clean triples
        graph = {}  # {(h, t): r}
        incorrectly_formatted_triples = []  # those triples that len(triples) != 3
        triples_wo_ner_entity = []  # those triples that have entities out of ner entities
        triple_tuples = []  # all clean triples

        # Step 1: process OpenIE results
        for row in tqdm(extracted_triples, total=len(extracted_triples)):
            ner_entities = [processing_phrases(p) for p in row["extracted_entities"]]
            triples = row["extracted_triples"]
            doc_json = row

            clean_triples = []
            unclean_triples = []
            doc_entities = set()  # clean entities related to each sample

            # Populate Triples from OpenIE
            for triple in triples:
                if not isinstance(triple, list) or any(
                    isinstance(i, list) or isinstance(i, tuple) for i in triple
                ):
                    continue

                if len(triple) > 1:
                    if len(triple) != 3:
                        clean_triple = [processing_phrases(p) for p in triple]
                        incorrectly_formatted_triples.append(triple)
                        unclean_triples.append(triple)
                    else:
                        clean_triple = [processing_phrases(p) for p in triple]
                        if "" in clean_triple:  # filter the triples with ''
                            incorrectly_formatted_triples.append(triple)  # modify
                            unclean_triples.append(triple)
                            continue

                        clean_triples.append(clean_triple)
                        phrases.extend(clean_triple)

                        head_ent = clean_triple[0]
                        tail_ent = clean_triple[2]

                        if (
                            head_ent not in ner_entities
                            and tail_ent not in ner_entities
                        ):
                            triples_wo_ner_entity.append(triple)

                        graph[(head_ent, tail_ent)] = clean_triple[1]

                        for triple_entity in [clean_triple[0], clean_triple[2]]:
                            entities.append(triple_entity)
                            doc_entities.add(triple_entity)

                doc_json["entities"] = list(set(doc_entities))
                doc_json["clean_triples"] = clean_triples
                doc_json["noisy_triples"] = unclean_triples
                triple_tuples.append(clean_triples)

                passage_json.append(doc_json)

        with open(os.path.join(self.tmp_dir, "passage_info.json"), "w") as f:
            json.dump(passage_json, f, indent=4)

        logging.info(f"Total number of processed data: {len(triple_tuples)}")

        lose_facts = []  # clean triples
        for triples in triple_tuples:
            lose_facts.extend([tuple(t) for t in triples])
        lose_fact_dict = {f: i for i, f in enumerate(lose_facts)}  # triples2id
        unique_phrases = list(np.unique(entities))  # Number of entities from documents
        unique_relations = np.unique(
            list(graph.values()) + ["equivalent"]
        )  # Number of relations from documents
        kb_phrase_dict = {p: i for i, p in enumerate(unique_phrases)}  # entities2id
        # Step 2: create raw graph
        logger.info("Creating Graph from OpenIE results")

        if self.cosine_sim_edges:
            self.augment_graph(
                graph, kb_phrase_dict=kb_phrase_dict
            )  # combine raw graph with synonyms edges

        synonymy_edges = {edge for edge in graph.keys() if graph[edge] == "equivalent"}
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
            ("Number of Edges", len(graph)),
            ("Number of Unique Entities", len(np.unique(entities))),
            ("Number of Synonymy Edges", len(synonymy_edges)),
            ("Number of Unique Relations", len(unique_relations)),
        ]

        logger.info("\n%s", pd.DataFrame(stat_df).set_index(0))

        return graph

    def augment_graph(self, graph: dict[Any, Any], kb_phrase_dict: dict) -> None:
        """
        Augment the graph with synonyms edges
        """
        logger.info("Augmenting graph from similarity")

        unique_phrases = list(kb_phrase_dict.keys())
        processed_phrases = [processing_phrases(p) for p in unique_phrases]

        self.el_model.index(processed_phrases)

        logger.info("Finding similar entities")
        sim_neighbors = self.el_model(processed_phrases, topk=self.max_sim_neighbors)

        logger.info("Adding synonymy edges")
        for phrase, neighbors in tqdm(sim_neighbors.items()):
            synonyms = []  # [(phrase_id, score)]
            if len(re.sub("[^A-Za-z0-9]", "", phrase)) > 2:
                phrase_id = kb_phrase_dict[phrase]
                if phrase_id is not None:
                    num_nns = 0
                    for neighbor in neighbors:
                        n_entity = neighbor["entity"]
                        n_score = neighbor["norm_score"]
                        if n_score < self.threshold or num_nns > self.max_sim_neighbors:
                            break
                        if n_entity != phrase:
                            phrase2_id = kb_phrase_dict[n_entity]
                            if phrase2_id is not None:
                                phrase2 = n_entity
                                synonyms.append((n_entity, n_score))
                                graph[(phrase, phrase2)] = "equivalent"
                                num_nns += 1
