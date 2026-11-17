

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class ComponentConfig:
    kind: str = "noop"
    provider: str = ""
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    adapter_path: str = ""
    prompt_key: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float = 60.0
    max_concurrency: int = 1
    retry_attempts: int = 1
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataConfig:
    dataset: str = ""
    train_jsonl: str = ""
    eval_jsonl: str = ""
    eval_in_domain_jsonl: str = ""
    eval_ood_jsonl: str = ""
    manifest: str = ""
    output: str = ""
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    stage: str = "sft"
    algorithm: str = "grpo"
    base_model: str = ""
    adapter_path: str = ""
    output_dir: str = ""
    run_id: str = ""
    prompt_key: str = "world_local"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant_with_warmup"
    seed: int = 42
    num_train_epochs: float = 1.0
    grpo_epochs: int = 1
    num_policy_epochs: int = 4
    group_size: int = 4
    clip_eps: float = 0.2
    gspo_clip_eps_low: float = 3e-4
    gspo_clip_eps_high: float = 4e-4
    kl_beta: float = 0.04
    temperature: float = 0.9
    top_p: float = 0.95
    max_grad_norm: float = 1.0
    per_device_batch_size: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    per_device_prompt_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_new_tokens: int | None = 6000
    max_length: int = 20000
    max_input_tokens: int | None = None
    max_output_tokens: int = 6000
    image_min_pixels: int = 256 * 28 * 28
    image_max_pixels: int = 1024 * 28 * 28
    min_supervised_tokens: int = 16
    assistant_loss_scope: str = "full"
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    max_current_md_chars: int | None = None
    max_assistant_output_chars: int | None = None
    device_mode: str = "auto"
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    dataloader_num_workers: int = 0
    dataloader_prefetch_factor: int | None = None
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    reward_embedding_model: str = ""
    element_count_free_ratio: float = 0.1
    element_count_anchor_ratio: float = 0.3
    element_count_anchor_multiplier: float = 0.8
    required_physical_gpus: str = ""
    dry_run: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    predictions_jsonl: str = ""
    output: str = ""
    prediction_field: str = "model_output"
    target_field: str = "target_next_md"
    use_embedding: bool = True
    embedding_model: str = ""
    dry_run: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "prediction"
    output_dir: str = ""
    dataset: str = ""
    split: str = ""
    max_tasks: int | None = None
    methods: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    dry_run: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    agent: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(
            kind="llm",
            model="Qwen3.5-35B",
            temperature=0.0,
            max_tokens=5000,
            options={
                "enable_thinking": True,
                "reasoning_effort": "medium",
                "thinking_token_budget": 4096,
            },
        )
    )
    global_world_model: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(
            kind="llm",
            model="Qwen3.5-9B",
            temperature=0.0,
            max_tokens=5000,
            options={"enable_thinking": False},
        )
    )
    local_world_model: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(
            kind="llm",
            model="Qwen3.5-9B",
            temperature=0.0,
            max_tokens=5000,
            options={"enable_thinking": False, "max_new_tokens": 5000},
        )
    )
    memory: ComponentConfig = field(default_factory=lambda: ComponentConfig(kind="noop"))
    environment: ComponentConfig = field(default_factory=lambda: ComponentConfig(kind="dry_run"))
    mode: str = "full"
    enable_global_wm: bool = True
    enable_local_wm: bool = True
    enable_memory: bool = False


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


T = TypeVar("T")


def _is_dataclass_type(type_hint: Any) -> bool:
    return isinstance(type_hint, type) and is_dataclass(type_hint)


def _coerce_value(type_hint: Any, value: Any) -> Any:
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    if _is_dataclass_type(type_hint):
        if not isinstance(value, Mapping):
            raise ValueError(f"Expected mapping for {type_hint.__name__}, got {type(value).__name__}")
        return dataclass_from_mapping(type_hint, value)
    if origin is tuple and args:
        if isinstance(value, str):
            raise ValueError(f"Expected sequence for {type_hint}, got string")
        return tuple(value or ())
    if origin is dict:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(f"Expected mapping for {type_hint}, got {type(value).__name__}")
        return dict(value or {})
    if origin is None:
        return value
    return value


def dataclass_from_mapping(cls: type[T], values: Mapping[str, Any]) -> T:
    known = {item.name: item for item in fields(cls)}
    type_hints = get_type_hints(cls)
    kwargs = {}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown config key for {cls.__name__}: {key}")
        kwargs[key] = _coerce_value(type_hints.get(key, known[key].type), value)
    return cls(**kwargs)


def load_run_config(path: Path | str | None) -> RunConfig:
    if path is None or str(path).strip() == "":
        return RunConfig()
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Run config JSON must contain an object")
    return dataclass_from_mapping(RunConfig, data)


def merge_run_config(config: RunConfig, overrides: Mapping[str, Any]) -> RunConfig:
    payload = asdict(config)
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        target = payload
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown config override namespace: {dotted_key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown config override key: {dotted_key}")
        target[parts[-1]] = value
    return dataclass_from_mapping(RunConfig, payload)


def provided_cli_dests(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> set[str]:
    tokens = sys.argv[1:] if argv is None else list(argv)
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    provided: set[str] = set()
    for token in tokens:
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            provided.add(option_to_dest[option])
    return provided


def apply_config_defaults(args: argparse.Namespace, values: Mapping[str, Any], provided_dests: set[str]) -> argparse.Namespace:
    for dest, value in values.items():
        if dest in provided_dests or value is None:
            continue
        setattr(args, dest, value)
    return args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect MKD2World explicit run configuration.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--print-default", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = RunConfig() if args.print_default else load_run_config(args.config)
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
