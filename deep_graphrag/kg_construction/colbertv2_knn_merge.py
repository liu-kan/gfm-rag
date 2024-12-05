# mypy: ignore-errors
import argparse
import os.path
import pickle

import pandas as pd
from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig

from .processing import processing_phrases


def retrieve_knn(kb, queries, dataset, duplicate=True, nns=100):
    checkpoint_path = "checkpoints/exp/colbertv2.0"

    if duplicate:
        kb = list(
            set(list(kb) + list(queries))
        )  # Duplicating queries to obtain score of query to query and normalize

    root = f"data/{dataset}/tmp_merge/lm_vectors/colbert/"
    if not os.path.exists(root):
        os.makedirs(root)

    with open(
        f"data/{dataset}/tmp_merge/lm_vectors/colbert/corpus.tsv", "w"
    ) as f:  # save to tsv
        for pid, p in enumerate(kb):
            f.write(f'{pid}\t"{p}"' + "\n")

    with open(
        f"data/{dataset}/tmp_merge/lm_vectors/colbert/queries.tsv", "w"
    ) as f:  # save to tsv
        for qid, q in enumerate(queries):
            f.write(f"{qid}\t{q}" + "\n")

    # index
    with Run().context(RunConfig(nranks=1, experiment="knn", root=root)):
        config = ColBERTConfig(nbits=2, root=root)
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(
            name="nbits_2",
            collection=f"data/{dataset}/tmp_merge/lm_vectors/colbert/corpus.tsv",
            overwrite=True,
        )

    # retrieval
    with Run().context(RunConfig(nranks=1, experiment="knn", root=root)):
        config = ColBERTConfig(
            root=root,
        )
        searcher = Searcher(index="nbits_2", config=config)
        queries = Queries(f"data/{dataset}/tmp_merge/lm_vectors/colbert/queries.tsv")
        ranking = searcher.search_all(queries, k=nns)

    ranking_dict = {}

    for i in range(len(queries)):
        query = queries[i]
        rank = ranking.data[i]
        max_score = rank[0][2]
        if duplicate:
            rank = rank[1:]
        ranking_dict[query] = (
            [kb[r[0]] for r in rank],
            [r[2] / max_score for r in rank],
        )

    return ranking_dict


def colbertv2_knn(dataset: str, filename: str) -> None:
    string_filename = f"data/{dataset}/tmp_merge/{filename}"
    # prepare tsv data
    string_df = pd.read_csv(string_filename, sep="\t")
    string_df.strings = [processing_phrases(str(s)) for s in string_df.strings]

    queries = string_df[string_df.type == "query"]
    kb = string_df[string_df.type == "kb"]

    nearest_neighbors = retrieve_knn(kb.strings.values, queries.strings.values, dataset)
    output_path = "data/{}/tmp_merge/lm_vectors/colbert/nearest_neighbor_{}.p".format(
        dataset, filename.split(".")[0]
    )
    pickle.dump(nearest_neighbors, open(output_path, "wb"))
    print(f"Saved nearest neighbors to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

    string_filename = f"data/{args.dataset}/tmp/{args.filename}"

    # prepare tsv data
    string_df = pd.read_csv(string_filename, sep="\t")
    string_df.strings = [processing_phrases(str(s)) for s in string_df.strings]

    queries = string_df[string_df.type == "query"]
    kb = string_df[string_df.type == "kb"]

    nearest_neighbors = retrieve_knn(kb.strings.values, queries.strings.values, args)
    output_path = "output/lm_vectors/colbert/{}/nearest_neighbor_{}.p".format(
        args.dataset, string_filename.split("/")[-1].split(".")[0]
    )
    pickle.dump(nearest_neighbors, open(output_path, "wb"))
    print(f"Saved nearest neighbors to {output_path}")
