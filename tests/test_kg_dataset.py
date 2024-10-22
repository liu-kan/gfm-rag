from deep_graphrag.datasets.kg_dataset import KGDataset


def test_kg_dataset() -> None:
    dataset = KGDataset(
        root="data",
        data_name="hotpotqa_example",
        text_emb_model_name="sentence-transformers/all-mpnet-base-v2",
    )
    kg = dataset[0]
    assert kg.num_nodes == 82157
    assert kg.num_relations == 35768


if __name__ == "__main__":
    test_kg_dataset()
