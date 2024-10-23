from deep_graphrag.datasets.kg_dataset import KGDataset
from deep_graphrag.datasets.qa_dataset import QADataset


def test_kg_dataset() -> None:
    dataset = KGDataset(
        root="data",
        data_name="hotpotqa_example",
        text_emb_model_name="sentence-transformers/all-mpnet-base-v2",
    )
    kg = dataset[0]
    assert kg.num_nodes == 85387
    assert kg.num_relations == 47438


def test_qa_dataset() -> None:
    dataset = QADataset(
        root="data",
        data_name="hotpotqa_example",
        text_emb_model_name="sentence-transformers/all-mpnet-base-v2",
    )
    assert len(dataset._data) == 1000


if __name__ == "__main__":
    # test_kg_dataset()
    test_qa_dataset()
