from dataclasses import dataclass, field
from typing import Optional, Union
from trl import SFTConfig, GRPOConfig

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the eval data."}
    )

@dataclass
class TRL_SFTTrainingArguments(SFTConfig):
    cache_dir: Optional[str] = field(default=None)
    # optim: str = field(default="adamw_torch")
    checkpoint: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."  # noqa: E501
        },
    )
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = field(
        default_factory={"use_reentrant": False}.copy,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )
    gradient_accumulation_steps: int = field(default=4) # Increase to 4 for smoother training

    # dataset_text_field = "text",
    per_device_train_batch_size: int = 1
    # gradient_accumulation_steps = 4 # Use GA to mimic batch size!
    warmup_steps: int = 5
    num_train_epochs: int = 1 # Set this for 1 full training run.
    # max_steps = 30,
    learning_rate: float = 5e-5 # Reduce to 2e-5 for long training runs
    logging_steps: int = 1
    optim: str= "adamw_torch_fused"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    report_to: str = "wandb" # Use this for WandB etc


@dataclass
class TRL_GRPOTrainingArguments(GRPOConfig):
    cache_dir: Optional[str] = field(default=None)
    # optim: str = field(default="adamw_torch")
    checkpoint: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."  # noqa: E501
        },
    )
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = field(
        default_factory={"use_reentrant": False}.copy,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )

    use_vllm: bool = field(default = True) # use vLLM for fast inference!
    vllm_mode: str = field(default = "server")
    # vllm_mode="colocate"
    learning_rate = 5e-5
    adam_beta1 = 0.9
    adam_beta2 = 0.99
    weight_decay = 0.1
    warmup_ratio = 0.1
    lr_scheduler_type = "cosine"
    optim = "adamw_torch_fused"
    logging_steps = 1
    bf16 = True
    fp16 = False
    per_device_train_batch_size = 1
    gradient_accumulation_steps: int = field(default=4) # Increase to 4 for smoother training
    evaluation_strategy="steps"
    num_generations = 4 # Decrease if out of memory
    max_prompt_length = 1024 + 512
    max_completion_length = 1024
    num_train_epochs = 1 # Set to 1 for a full training run
    # max_steps = 250,
    save_steps = 250
    max_grad_norm = 0.1
    max_grad_norm = 1.0
    report_to: str = "wandb" # Use this for WandB etc
    output_dir = "outputs"
# )
