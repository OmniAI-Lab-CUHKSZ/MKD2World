"""Fine-tune Qwen3.5-35B-A3B with LoRA to reconstruct the current screenshot as TSM.

Launch with two processes under FSDP2:

    torchrun --standalone --nproc_per_node=2 -m mkd2world.training.qwen35_35b_current_tsm \
      --train-jsonl /path/to/current_tsm.jsonl --base-model /path/to/Qwen3.5-35B-A3B --output-dir /path/to/out
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

FSDP_WRAP_CLASSES = ('Qwen3_5MoeDecoderLayer', 'Qwen3_5MoeVisionBlock')
LANGUAGE_SUFFIXES = ('.q_proj', '.k_proj', '.v_proj', '.o_proj', '.in_proj_qkv', '.in_proj_z', '.in_proj_a', '.in_proj_b', '.out_proj', '.gate_proj', '.up_proj', '.down_proj')
VISION_SUFFIXES = ('.attn.qkv', '.attn.proj', '.mlp.linear_fc1', '.mlp.linear_fc2', '.merger.linear_fc1', '.merger.linear_fc2')
MAX_SUPERVISED_TOKENS = 6000


def exposure_order(unique_rows: int, *, max_steps: int, global_batch: int) -> list[int]:
    """One pass in source order, then wrap around to fill the last global batch."""
    required = max_steps * global_batch
    return [index % unique_rows for index in range(required)]


def local_exposure_indices(ordered_indices: Sequence[int], *, rank: int, world_size: int, gradient_accumulation_steps: int) -> list[int]:
    """Give each rank its contiguous slice of every global batch."""
    global_batch = world_size * gradient_accumulation_steps
    selected: list[int] = []
    for step_start in range(0, len(ordered_indices), global_batch):
        block = ordered_indices[step_start:step_start + global_batch]
        start = rank * gradient_accumulation_steps
        selected.extend(block[start:start + gradient_accumulation_steps])
    return selected


def expanded_qwen35_lora_target_modules(model: Any) -> list[str]:
    selected: list[str] = []
    for (name, module) in model.named_modules():
        weight = getattr(module, 'weight', None)
        if weight is None or getattr(weight, 'ndim', None) != 2:
            continue
        is_language = 'language_model.layers.' in name and name.endswith(LANGUAGE_SUFFIXES)
        is_vision = '.visual.' in f'.{name}.' and name.endswith(VISION_SUFFIXES)
        if is_language or is_vision:
            selected.append(name)
    return sorted(set(selected))


def _parse_meta(row: Mapping[str, Any]) -> Mapping[str, Any]:
    meta: Any = row.get('meta')
    if isinstance(meta, str):
        meta = json.loads(meta)
    return meta if isinstance(meta, Mapping) else {}


def sanitize_current_training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the screenshot and the current-page TSM target; the input carries no Markdown."""
    meta = _parse_meta(row)
    target = str(row.get('target_next_md') or '')
    image_path = row.get('current_image_path') or row.get('image_path')
    identifier = row.get('id') or row.get('sample_id')
    if row.get('current_md') != '' or not target or image_path in (None, '') or identifier in (None, ''):
        raise ValueError(f'row {identifier!r} is not a current-page row with an empty current_md, a target, and an image')
    return {'id': str(identifier), 'task_type': 'localwm_cur', 'current_image_path': str(image_path), 'current_md': '', 'target_next_md': target, 'assistant_output': target, 'action_analysis': '', 'meta': dict(meta), 'meta_json': json.dumps(meta, ensure_ascii=False, sort_keys=True)}


class CurrentJsonlDataset:
    """Rows whose ``meta.state_side`` is ``current``, read by byte offset in a given order."""

    def __init__(self, path: Path, *, max_rows: int | None) -> None:
        self.path = path
        self.offsets: list[int] = []
        with path.open('rb') as handle:
            while max_rows is None or len(self.offsets) < max_rows:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip() and str(_parse_meta(json.loads(line)).get('state_side') or '') == 'current':
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f'no current-page rows in {path}')
        self.indices = list(range(len(self.offsets)))
        self._handle = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        if self._handle is None:
            self._handle = self.path.open('r', encoding='utf-8')
        self._handle.seek(self.offsets[self.indices[position]])
        return sanitize_current_training_row(json.loads(self._handle.readline()))

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__) | {'_handle': None}


def build_training_arguments(TrainingArguments: Any, args: argparse.Namespace, checkpoint_root: Path) -> Any:
    return TrainingArguments(output_dir=str(checkpoint_root), per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps, max_steps=args.max_steps, learning_rate=args.learning_rate, lr_scheduler_type='cosine', warmup_ratio=0.03, weight_decay=0.01, optim='adamw_torch_fused', adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, seed=0, data_seed=0, bf16=True, fp16=False, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False, 'preserve_rng_state': True}, fsdp=True, fsdp_config={'version': 2, 'reshard_after_forward': True, 'cpu_offload': False, 'activation_checkpointing': False, 'cpu_ram_efficient_loading': True, 'state_dict_type': 'SHARDED_STATE_DICT', 'auto_wrap_policy': 'TRANSFORMER_BASED_WRAP', 'transformer_layer_cls_to_wrap': list(FSDP_WRAP_CLASSES)}, save_strategy='steps', save_steps=args.save_steps, save_total_limit=None, save_only_model=False, logging_steps=1, eval_strategy='no', remove_unused_columns=False, label_names=['labels'], report_to=[], dataloader_num_workers=0, dataloader_pin_memory=True, dataloader_drop_last=False, ignore_data_skip=True, average_tokens_across_devices=True, disable_tqdm=True)


def build_selective_trainer_class(Trainer: Any, torch_module: Any) -> type:
    functional = torch_module.nn.functional
    DataLoader = torch_module.utils.data.DataLoader

    class SelectiveTokenTrainer(Trainer):
        """Compute logits only at supervised positions; normalize by the all-rank token count."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = True

        def get_train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=1, shuffle=False, collate_fn=self.data_collator, num_workers=0, pin_memory=True, drop_last=False)

        def compute_loss(self, model: Any, inputs: Mapping[str, Any], return_outputs: bool=False, num_items_in_batch: Any | None=None):
            model_inputs = dict(inputs)
            labels = model_inputs.pop('labels')
            target_positions = torch_module.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten() + 1
            if target_positions.numel() < 1 or target_positions.numel() > MAX_SUPERVISED_TOKENS:
                raise ValueError(f'supervised shifted tokens={target_positions.numel()}, expected 1..{MAX_SUPERVISED_TOKENS}')
            outputs = model(**model_inputs, labels=None, logits_to_keep=target_positions - 1)
            logits = outputs.logits
            targets = labels[0, target_positions].to(logits.device)
            local_sum = functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='sum')
            loss = local_sum / torch_module.as_tensor(num_items_in_batch, device=local_sum.device, dtype=local_sum.dtype)
            if self.args.average_tokens_across_devices:
                loss = loss * self.accelerator.num_processes
            return (loss, outputs) if return_outputs else loss
    return SelectiveTokenTrainer


def build_adapter_export_callback(TrainerCallback: Any, *, torch_module: Any, checkpoint_root: Path, max_steps: int, rank: int) -> Any:

    class AdapterExportCallback(TrainerCallback):
        """Save at the last step too, and write a standard PEFT adapter next to each FSDP checkpoint."""

        def on_step_end(self, args, state, control, **kwargs):
            if int(state.global_step) >= max_steps:
                control.should_save = True
            return control

        def on_save(self, args, state, control, model=None, **kwargs):
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            state_dict = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True))
            if rank == 0:
                model.save_pretrained(str(checkpoint_root / f'checkpoint-{int(state.global_step)}'), state_dict=state_dict, safe_serialization=True)
            if torch_module.distributed.is_available() and torch_module.distributed.is_initialized():
                torch_module.distributed.barrier()
            return control
    return AdapterExportCallback()


def run_training(args: argparse.Namespace) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainingArguments
    from mkd2world.training.sft import LocalWMSftCollator
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    checkpoint_root = args.output_dir.resolve() / 'checkpoints'
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    training_args = build_training_arguments(TrainingArguments, args, checkpoint_root)
    dataset = CurrentJsonlDataset(args.train_jsonl, max_rows=args.max_rows)
    ordered = exposure_order(len(dataset.offsets), max_steps=args.max_steps, global_batch=world_size * args.gradient_accumulation_steps)
    dataset.indices = local_exposure_indices(ordered, rank=rank, world_size=world_size, gradient_accumulation_steps=args.gradient_accumulation_steps)
    processor = AutoProcessor.from_pretrained(args.base_model, min_pixels=200704, max_pixels=802816, trust_remote_code=True)
    collator = LocalWMSftCollator(processor=processor, image_module=Image, max_length=20000, max_output_tokens=6000, assistant_loss_scope='full', min_supervised_tokens=16, prompt_key='localwm_cur')
    model = AutoModelForImageTextToText.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map=None, attn_implementation='sdpa', trust_remote_code=True)
    model.config.use_cache = False
    model.config.output_router_logits = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=expanded_qwen35_lora_target_modules(model)))
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    if rank == 0:
        model.print_trainable_parameters()
    callback = build_adapter_export_callback(TrainerCallback, torch_module=torch, checkpoint_root=checkpoint_root, max_steps=args.max_steps, rank=rank)
    trainer = build_selective_trainer_class(Trainer, torch)(model=model, args=training_args, train_dataset=dataset, data_collator=collator, callbacks=[callback])
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    return checkpoint_root / f'checkpoint-{int(trainer.state.global_step)}'


def parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-jsonl', type=Path, required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-rows', type=int, default=50000)
    parser.add_argument('--max-steps', type=int, default=834)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=2e-06)
    parser.add_argument('--save-steps', type=int, default=200)
    parser.add_argument('--resume-from-checkpoint', type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None=None) -> None:
    final_checkpoint = run_training(parse_args(argv))
    if int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps({'final_checkpoint': str(final_checkpoint)}))


if __name__ == '__main__':
    main()
