import transformers

from llm_finetune.arguments import (
    ModelArguments,
    DataArguments,
    TRL_SFTTrainingArguments,
)
from llm_finetune.dataset import make_supervised_data_module_trl

from trl import SFTTrainer
from peft import get_peft_model, LoraConfig



def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TRL_SFTTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args, 'training_args')

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    

    # Create a LoRA configuration object
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",  # Adding task type for language models
        # init_lora_weights="gaussian",  # Initializing weights
        # random_seed=3407  # Changed from random_state to random_seed
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    data_module = make_supervised_data_module_trl(
        data_args=data_args,
    )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )
    trainer.train(training_args.checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
