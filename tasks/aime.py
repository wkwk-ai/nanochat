import re
from datasets import load_dataset
from tasks.common import Task

# Match \boxed{123} or \boxed{-45}
BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")

def extract_boxed_answer(text):
    """
    Extract content inside \\boxed{...}
    Returns stripped string, or None if not found.
    """
    if text is None:
        return None
    match = BOX_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


class AIME(Task):
    """
    AIME 2024 / 2025 evaluation task.
    """

    def __init__(self, year, split="test", **kwargs):
        super().__init__(**kwargs)
        assert year in [24, 25], "year must be 24 or 25"

        self.year = year
        dataset_name = f"math-ai/aime{year}"
        self.ds = load_dataset(dataset_name, split=split)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):

        row = self.ds[index]

        problem = row["problem"]

        # Ground-truth answer format differs by year
        if self.year == 24:

            answer = row["solution"]
        else:

            answer = row["answer"]

        formatted_question = (
            "Problem:\n"
            f"{problem}\n\n"
            "Mark your solution with \\boxed\n"
            "Answer:"
        )

        messages = [
            {"role": "user", "content": formatted_question},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(answer)}
                ],
            },
        ]

        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):

        assert isinstance(assistant_response, str), \
            "Assuming simple string response for now"

        assistant_message = conversation["messages"][-1]
        assert assistant_message["role"] == "assistant"
        assert isinstance(assistant_message["content"], list)

        gt_text = assistant_message["content"][-1]["text"]

        # Extract ground-truth answer
        if self.year == 24:
            ref_ans = extract_boxed_answer(gt_text)
        else:
            ref_ans = gt_text.strip()

        # Extract predicted answer (always boxed)
        pred_ans = extract_boxed_answer(assistant_response)

        return int(
            ref_ans is not None and
            pred_ans is not None and
            pred_ans == ref_ans
        )

    def reward(self, conversation, assistant_response):
        """
        Same as evaluate, used during RL.
        """
        return float(self.evaluate(conversation, assistant_response))
