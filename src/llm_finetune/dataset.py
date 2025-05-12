from datasets import load_from_disk
from .grpo import process_agnews_dataset_for_reasoning



def get_agnews_questions_for_sft(dataset):
  dataset = dataset.map(lambda x: { # type: ignore
      'prompt': [
          {'role': 'user', 'content': x['input_train']}
      ],
      'completion': [
          {'role': 'assistant', 'content': x['answer']}
      ],
  }) # type: ignore
  return dataset # type: ignore

def make_supervised_data_module_trl(data_args):
    """(New Method) Make dataset for TRL supervised fine-tuning."""

    train_dataset = get_agnews_questions_for_sft(load_from_disk(data_args.train_data_path))
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = get_agnews_questions_for_sft(load_from_disk(data_args.eval_data_path))
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

def make_reasoning_data_module_trl(data_args):
    """(New Method) Make dataset for TRL supervised fine-tuning."""

    train_dataset = process_agnews_dataset_for_reasoning(load_from_disk(data_args.train_data_path))
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = process_agnews_dataset_for_reasoning(load_from_disk(data_args.eval_data_path))
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )