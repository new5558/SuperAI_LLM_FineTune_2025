import transformers

from llm_finetune.arguments import (
    ModelArguments,
    DataArguments,
    TRL_GRPOTrainingArguments,
)
from llm_finetune.dataset import make_reasoning_data_module_trl
from llm_finetune.grpo import (
    xmlcount_reward_func,
    int_reward_func,
    soft_format_reward_func,
    correctness_reward_func,
    reasoning_length_reward
)

from trl import GRPOTrainer



def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TRL_GRPOTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args, 'training_args')

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    data_module = make_reasoning_data_module_trl(
        data_args=data_args,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            int_reward_func,
            reasoning_length_reward,
            correctness_reward_func,
        ],
        **data_module
    )
    trainer.train(training_args.checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
