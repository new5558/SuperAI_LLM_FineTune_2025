import re
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT_REASONING = """
We want to check classfy the news

News consists of class ids 1-4 where 1-World, 2-Sports, 3-Business, 4-Sci/Tech

answer in number '0' '1' '2' '3' in <answer>, think in the <think> tag before answer
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_xml_think(text: str) -> str:
    answer = text.split("<think>")[-1]
    answer = answer.split("</think>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def process_agnews_dataset_for_reasoning(dataset):
  dataset = dataset.map(lambda x: { # type: ignore
      'prompt': [
          {'role': 'system', 'content': SYSTEM_PROMPT_REASONING},
          {'role': 'user', 'content': x['input_train']}
      ],
  }) # type: ignore
  return dataset # type: ignore

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def reasoning_length_reward(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    think_responses_len = [len(extract_xml_think(r)) for r in responses]
    return [0.5 if (L >= 30 and L <= 2000) else -0.5 for L in think_responses_len]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>(?:.|\n)*?</think>\s*<answer>(?:.|\n)*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]