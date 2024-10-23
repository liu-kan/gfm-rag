from deep_graphrag.datasets.kg_dataset import KGDataset
from deep_graphrag.datasets.qa_dataset import QADataset


def test_kg_dataset() -> None:
    dataset = KGDataset(
        root="data",
        data_name="hotpotqa_example",
        text_emb_model_name="sentence-transformers/all-mpnet-base-v2",
    )
    kg = dataset[0]
    assert kg.num_nodes == 82157
    assert kg.num_relations == 35768


def test_qa_dataset() -> None:
    dataset = QADataset(
        root="data",
        data_name="hotpotqa_example",
        text_emb_model_name="sentence-transformers/all-mpnet-base-v2",
    )
    train_data, test_data = dataset._data
    assert len(train_data) == 800
    assert len(test_data) == 200


if __name__ == "__main__":
    test_kg_dataset()
    test_qa_dataset()
