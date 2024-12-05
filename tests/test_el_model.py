def test_colbert_el_model() -> None:
    import json

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "deep_graphrag.kg_construction.entity_linking_model.ColbertELModel",
            "checkpint_path": "tmp/colbertv2.0",
            "root": "tmp",
        }
    )

    el_model = instantiate(cfg)
    ner_entity_list = ["south chicago community hospital", "july 13 14  1966"]
    with open(
        "data/hotpotqa_test/processed/stage2/sentence-transformers_all-mpnet-base-v2/ent2id.json"
    ) as fin:
        ent2id = json.load(fin)
    entity_list = list(ent2id.keys())
    el_model.index(entity_list)
    linked_entity_list = el_model(ner_entity_list)
    print(linked_entity_list)
    assert isinstance(linked_entity_list, list)


def test_dpr_el_model() -> None:
    import json

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "deep_graphrag.kg_construction.entity_linking_model.DPRELModel",
            "model_name": "BAAI/bge-large-en-v1.5",
            "root": "tmp",
            "use_cache": True,
            "normalize": True,
        }
    )

    el_model = instantiate(cfg)
    ner_entity_list = ["south chicago community hospital", "july 13 14  1966"]
    with open(
        "data/hotpotqa_test/processed/stage2/sentence-transformers_all-mpnet-base-v2/ent2id.json"
    ) as fin:
        ent2id = json.load(fin)
    entity_list = list(ent2id.keys())
    el_model.index(entity_list)
    linked_entity_list = el_model(ner_entity_list)
    print(linked_entity_list)
    assert isinstance(linked_entity_list, list)


if __name__ == "__main__":
    # test_colbert_el_model()
    test_dpr_el_model()
