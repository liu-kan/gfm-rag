from omegaconf import DictConfig


class QAPromptBuilder:
    def __init__(self, prompt_cfg: DictConfig) -> None:
        self.cfg = prompt_cfg
        self.system_prompt = self.cfg.system_prompt
        self.doc_prompt = self.cfg.doc_prompt
        self.question_prompt = self.cfg.question_prompt
        self.examples = self.cfg.examples

    def build_input_prompt(self, question: str, retrieved_docs: list) -> list:
        prompt = [
            {"role": "system", "content": self.system_prompt},
        ]

        doc_context = "\n".join(
            [
                self.doc_prompt.format(title=doc["title"], content=doc["content"])
                for doc in retrieved_docs
            ]
        )

        question = self.question_prompt.format(question=question)

        if len(self.examples) > 0:
            for example in self.examples:
                prompt.extend(
                    [
                        {"role": "user", "content": example["input"]},
                        {"role": "assistant", "content": example["response"]},
                    ]
                )
        prompt.append(
            {
                "role": "user",
                "content": doc_context + "\n" + question,
            }
        )

        return prompt
