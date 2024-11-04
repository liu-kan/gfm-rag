import json
import logging
import os
from multiprocessing.dummy import Pool as ThreadPool

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils import data as torch_data
from torch.utils.data import Dataset
from tqdm import tqdm

from deep_graphrag import utils
from deep_graphrag.datasets import QADataset
from deep_graphrag.ultra import query_utils

# A logger for this file
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = 'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'

DOC_PROMPT = "Wikipedia Title: {title}\n{content}\n"

ONE_SHOT_EXAMPLE = """Wikipedia Title: Milk and Honey (album)
Milk and Honey is an album by John Lennon and Yoko Ono released in 1984. Following the compilation "The John Lennon Collection", it is Lennon's eighth and final studio album, and the first posthumous release of new Lennon music, having been recorded in the last months of his life during and following the sessions for their 1980 album "Double Fantasy". It was assembled by Yoko Ono in association with the Geffen label.

Wikipedia Title: John Lennon Museum
John Lennon Museum (ジョン・レノン・ミュージアム , Jon Renon Myūjiamu ) was a museum located inside the Saitama Super Arena in Chūō-ku, Saitama, Saitama Prefecture, Japan. It was established to preserve knowledge of John Lennon's life and musical career. It displayed Lennon's widow Yoko Ono's collection of his memorabilia as well as other displays. The museum opened on October 9, 2000, the 60th anniversary of Lennon’s birth, and closed on September 30, 2010, when its exhibit contract with Yoko Ono expired. A tour of the museum began with a welcoming message and short film narrated by Yoko Ono (in Japanese with English headphones available), and ended at an avant-garde styled "reflection room" full of chairs facing a slide show of moving words and images. After this room there was a gift shop with John Lennon memorabilia available.

Wikipedia Title: Walls and Bridges
Walls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.

Wikipedia Title: Nobody Loves You (When You're Down and Out)
"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".

Wikipedia Title: Give Peace a Chance
"Give Peace a Chance" is an anti-war song written by John Lennon (credited to Lennon–McCartney), and performed with Yoko Ono in Montreal, Quebec, Canada. Released as a single in 1969 by the Plastic Ono Band on Apple Records (catalogue Apple 13 in the United Kingdom, Apple 1809 in the United States), it is the first solo single issued by Lennon, released when he was still a member of the Beatles, and became an anthem of the American anti-war movement during the 1970s. It peaked at number 14 on the "Billboard" Hot 100 and number 2 on the British singles chart.

Question: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?
Thought: """
ONE_SHOT_RESPONSE = """ The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album.
Answer: Walls and Bridges."""

ONE_SHOT_PROMPT = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "ONE_SHOT_EXAMPLE"},
    {"role": "assistant", "content": ONE_SHOT_RESPONSE},
]

QUESTION_PROMPT = "Question: {question}\nThought: "


@torch.no_grad()
def doc_retrieval(
    cfg: DictConfig,
    model: nn.Module,
    qa_data: Dataset,
    device: torch.device,
) -> list[dict]:
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    _, test_data = qa_data._data
    graph = qa_data.kg
    ent2docs = qa_data.ent2docs

    # Retrieve the supporting documents for each query
    sampler = torch_data.DistributedSampler(test_data, world_size, rank, shuffle=False)
    test_loader = torch_data.DataLoader(
        test_data, cfg.test.retrieval_batch_size, sampler=sampler
    )

    model.eval()
    all_predictions: list[dict] = []
    for batch in tqdm(test_loader):
        batch = query_utils.cuda(batch, device=device)
        ent_pred = model(graph, batch)
        doc_pred = torch.sparse.mm(ent_pred, ent2docs)
        idx = batch[4]
        all_predictions.extend(
            {"id": i, "ent_pred": e, "doc_pred": d}
            for i, e, d in zip(idx.cpu(), ent_pred.cpu(), doc_pred.cpu())
        )

    # Gather the predictions across all processes
    if utils.get_world_size() > 1:
        gathered_predictions = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_predictions, all_predictions)
    else:
        gathered_predictions = [all_predictions]  # type: ignore

    sorted_predictions = sorted(
        [item for sublist in gathered_predictions for item in sublist],  # type: ignore
        key=lambda x: x["id"],
    )
    utils.synchronize()
    return sorted_predictions


def ans_prediction(
    cfg: DictConfig, output_dir: str, qa_data: Dataset, retrieval_result: list[dict]
) -> None:
    llm = instantiate(cfg.llm)
    doc_retriever = get_class(cfg.retriever.doc_retriever)(qa_data.doc, qa_data.id2doc)
    test_data = qa_data.raw_test_data

    def predict(qa_input: tuple[dict, torch.Tensor]) -> dict | Exception:
        data, retrieval_doc = qa_input
        retrieved_docs = doc_retriever(retrieval_doc["doc_pred"], top_k=cfg.test.top_k)
        doc_context = "\n".join(
            [
                DOC_PROMPT.format(title=doc["title"], content=" ".join(doc["content"]))
                for doc in retrieved_docs
            ]
        )
        question = QUESTION_PROMPT.format(question=data["question"])

        if cfg.test.prompt_mode == "one-shot":
            message = ONE_SHOT_PROMPT + [
                {
                    "role": "user",
                    "content": doc_context + "\n\n" + question,
                }
            ]
        else:
            message = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": doc_context + "\n\n" + question},
            ]

        response = llm.generate_sentence(message)
        if isinstance(response, Exception):
            return response
        else:
            return {
                "id": data["id"],
                "question": data["question"],
                "answer": data["answer"],
                "response": response,
                "retrieved_docs": retrieved_docs,
            }

    with open(os.path.join(output_dir, "prediction.jsonl"), "w") as f:
        with ThreadPool(cfg.test.n_threads) as pool:
            for results in tqdm(
                pool.imap(predict, zip(test_data, retrieval_result)),
                total=len(test_data),
            ):
                if isinstance(results, Exception):
                    logger.error(f"Error: {results}")
                    continue

                f.write(json.dumps(results) + "\n")
                f.flush()

    # Evaluation
    evaluator = get_class(cfg.test.evaluator["_target_"])(
        prediction_file=os.path.join(output_dir, "prediction.jsonl")
    )
    metrics = evaluator.evaluate()
    query_utils.print_metrics(metrics, logger)
    return metrics


@hydra.main(config_path="config", config_name="stage3_qa_inference", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    utils.init_distributed_mode()
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")

    model, config = utils.load_model_from_pretrained(cfg.retriever.model_path)
    qa_data = QADataset(text_emb_model_name=config["text_emb_model"], **cfg.dataset)
    device = utils.get_device()
    model = model.to(device)

    qa_data.kg = qa_data.kg.to(device)
    qa_data.ent2docs = qa_data.ent2docs.to(device)

    retrieval_result = doc_retrieval(cfg, model, qa_data, device=device)
    if utils.is_main_process():
        torch.save(retrieval_result, os.path.join(output_dir, "retrieval_result.pt"))
        logger.info("Ranking saved to disk")
        ans_prediction(cfg, output_dir, qa_data, retrieval_result)


if __name__ == "__main__":
    main()
