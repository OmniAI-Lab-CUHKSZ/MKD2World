from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
EXPECTED_LORA_MODULES = 420
EXPECTED_LORA_TENSORS = 840
EXPECTED_LORA_PARAMETERS = 58240000
EXPECTED_TARGET_NAMES_SHA256 = 'ad0fb1298ade107841a7429f0988761764d68b47198a5e38984030cc5428ae3b'
EXPECTED_WORLD_SIZE = 2
EXPECTED_GRADIENT_ACCUMULATION = 30
EXPECTED_GLOBAL_BATCH = 60
EXPECTED_MAX_STEPS = 834
EXPECTED_MAX_SUPERVISED_TOKENS = 6000
CERTIFICATION_MIN_SUPERVISED_TOKENS = 5900
CERTIFICATION_IMAGE_WIDTH = 896
CERTIFICATION_IMAGE_HEIGHT = 896
CERTIFICATION_IMAGE_PIXELS = 802816
CHECKPOINT_STEPS = (200, 400, 600, 800, 834)
FORMAL_EVALUATION_FILES = ('scripts/evaluation/build_qwen35b_forgetting_probe.py', 'scripts/evaluation/qwen35b_forgetting_common.py', 'scripts/evaluation/evaluate_qwen35b_forgetting_gate.py', 'scripts/evaluation/qwen35b_forgetting_probe_manifest.schema.json', 'scripts/evaluation/evaluate_qwen35b_description_current.py', 'scripts/evaluation/qwen35b_description_current_assets.json')
FSDP_WRAP_CLASSES = ('Qwen3_5MoeDecoderLayer', 'Qwen3_5MoeVisionBlock')
LANGUAGE_SUFFIXES = ('.q_proj', '.k_proj', '.v_proj', '.o_proj', '.in_proj_qkv', '.in_proj_z', '.in_proj_a', '.in_proj_b', '.out_proj', '.gate_proj', '.up_proj', '.down_proj')
VISION_SUFFIXES = ('.attn.qkv', '.attn.proj', '.mlp.linear_fc1', '.mlp.linear_fc2', '.merger.linear_fc1', '.merger.linear_fc2')

class ContractError(RuntimeError):
    pass

@dataclass(frozen=True)
class AdapterGraphContract:
    module_count: int = EXPECTED_LORA_MODULES
    tensor_count: int = EXPECTED_LORA_TENSORS
    parameter_count: int = EXPECTED_LORA_PARAMETERS
    target_names_sha256: str = EXPECTED_TARGET_NAMES_SHA256
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def ordered_values_sha256(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()

def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)

def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ContractError(f'expected a JSON object: {path}')
    return value

def resolve_project_path(project_root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path

def validate_manifest_contract(manifest: Mapping[str, Any], resource: Mapping[str, Any]) -> None:
    failures: list[str] = []
    training = manifest.get('training', {})
    model = manifest.get('model', {})
    lora = model.get('lora', {}) if isinstance(model, Mapping) else {}
    data = manifest.get('data', {})
    current50k = data.get('current50k', {}) if isinstance(data, Mapping) else {}
    filtering = data.get('filter', {}) if isinstance(data, Mapping) else {}
    checkpoint = manifest.get('checkpoint', {})
    distributed = resource.get('distributed', {})
    activation = resource.get('activation_checkpointing', {})
    software = resource.get('software', {})
    original_sums = Path(str(model.get('original_sha256sums_path', '')))
    canonical_sums = Path(str(model.get('sha256sums_path', '')))
    canonical_receipt = Path(str(model.get('sha256sums_canonicalization_receipt_path', '')))
    expected_persistent_tmux = {'executable': {'path': 'MKD2WORLD_EXTERNAL_ROOT/envs/bin/tmux', 'sha256': '6684c9b0bd4af08461f9e476e0abee9c3f08daa5d55ed6fb7c663c000e09f83d', 'version': 'tmux 3.2a'}, 'libraries': [{'path': 'MKD2WORLD_EXTERNAL_ROOT/envs/bin/libutempter.so.0', 'sha256': 'd55f4fd318d57f674c2572cb45cd93f7b6f964409421e03a6186a340919f4c39'}, {'path': 'MKD2WORLD_EXTERNAL_ROOT/envs/bin/libevent_core-2.1.so.7', 'sha256': '0b7a5ee57efae362e849fee23518023b48af19f07aec31de6294e19b9ab7e5c1'}, {'path': 'MKD2WORLD_EXTERNAL_ROOT/envs/bin/libtinfo.so.6', 'sha256': 'bcd7edc10f0e3df476c0738280cd60c751c11b5b99fa13ded70f5439cb3a8ce9'}]}
    toolchain = software.get('persistent_cpp_toolchain', {})
    cuda_toolchain = software.get('persistent_cuda_toolchain', {})
    jit_cache = software.get('jit_cache', {})
    toolchain_root = 'MKD2WORLD_EXTERNAL_ROOT/runtime/toolchains/gxx-11.4.0-ubuntu22.04-v1'
    compiler_path = f'{toolchain_root}/usr/bin/g++-11'
    c_compiler_path = f'{toolchain_root}/usr/bin/gcc-11'
    cuda_toolchain_root = 'MKD2WORLD_EXTERNAL_ROOT/runtime/cuda-toolchains/cuda-13.0.88-cu130-v1'
    cuda_home = f'{cuda_toolchain_root}/site-packages/nvidia/cu13'
    nvcc_path = f'{cuda_home}/bin/nvcc'
    cache_root = 'MKD2WORLD_EXTERNAL_ROOT/runtime/kernel_cache/guiagent_qwen35b_v3_cur_gxx11_cuda130_v1'
    expected_toolchain_environment = {'CC': c_compiler_path, 'CXX': compiler_path, 'NVCC_CCBIN': compiler_path, 'COMPILER_PATH_prefix': [f'{toolchain_root}/usr/lib/gcc/x86_64-linux-gnu/11'], 'CPLUS_INCLUDE_PATH_prefix': [f'{toolchain_root}/usr/include/c++/11', f'{toolchain_root}/usr/include/x86_64-linux-gnu/c++/11', f'{toolchain_root}/usr/include/c++/11/backward']}
    expected_jit_cache = {'schema_version': 'guiagent-persistent-jit-cache-v1', 'root': cache_root, 'directories': {'TILELANG_CACHE_DIR': f'{cache_root}/tilelang', 'TILELANG_TMP_DIR': f'{cache_root}/tilelang/tmp', 'TORCH_EXTENSIONS_DIR': f'{cache_root}/torch_extensions', 'TRITON_CACHE_DIR': f'{cache_root}/triton', 'TORCHINDUCTOR_CACHE_DIR': f'{cache_root}/torchinductor', 'CUDA_CACHE_PATH': f'{cache_root}/cuda'}, 'fixed_environment': {'TILELANG_KERNEL_CACHE_USE_LIB_STAMP': '1'}, 'project_persistent': True, 'writable_probe_required': True}
    expected_scalars = {'training.world_size': (training.get('world_size'), EXPECTED_WORLD_SIZE), 'training.per_device_train_batch_size': (training.get('per_device_train_batch_size'), 1), 'training.gradient_accumulation_steps': (training.get('gradient_accumulation_steps'), EXPECTED_GRADIENT_ACCUMULATION), 'training.global_batch_size': (training.get('global_batch_size'), EXPECTED_GLOBAL_BATCH), 'training.max_steps': (training.get('max_steps'), EXPECTED_MAX_STEPS), 'training.learning_rate': (training.get('learning_rate'), 2e-06), 'training.max_length': (training.get('max_length'), 20000), 'training.max_output_tokens': (training.get('max_output_tokens'), EXPECTED_MAX_SUPERVISED_TOKENS), 'lora.modules': (lora.get('expected_modules'), EXPECTED_LORA_MODULES), 'lora.tensors': (lora.get('expected_tensors'), EXPECTED_LORA_TENSORS), 'lora.parameters': (lora.get('expected_parameters'), EXPECTED_LORA_PARAMETERS), 'current50k.rows': (current50k.get('rows'), 50000), 'current50k.domains': (current50k.get('unique_domains'), 854), 'current50k.image_list_path': (current50k.get('image_list_path'), 'dataset/GUIAgent/qwen35b_v3_cur_current50k_v1/image_paths.txt'), 'current50k.image_path_contract': (current50k.get('image_path_contract'), 'absolute_project_path_containing_render/images'), 'filter.field': (filtering.get('field'), 'meta.state_side'), 'filter.equals': (filtering.get('equals'), 'current'), 'fsdp.version': (distributed.get('fsdp_version'), 2), 'fsdp.world_size': (distributed.get('world_size'), 2), 'fsdp.state_dict_type': (distributed.get('state_dict_type'), 'SHARDED_STATE_DICT')}
    for (name, (actual, expected)) in expected_scalars.items():
        if actual != expected:
            failures.append(f'{name}={actual!r}, expected {expected!r}')
    calculated_batch = int(training.get('world_size', 0)) * int(training.get('per_device_train_batch_size', 0)) * int(training.get('gradient_accumulation_steps', 0))
    if calculated_batch != EXPECTED_GLOBAL_BATCH:
        failures.append(f'global batch formula produced {calculated_batch}, expected 60')
    if tuple(checkpoint.get('steps', ())) != CHECKPOINT_STEPS:
        failures.append('checkpoint.steps must be [200,400,600,800,834]')
    if checkpoint.get('save_strategy') != 'segment_end_only' or checkpoint.get('save_steps') is not None:
        failures.append('formal checkpoints must be callback-forced segment ends only')
    evaluation = manifest.get('evaluation', {})
    if evaluation.get('base_step_0_once') is not True:
        failures.append('bare-base step-0 evaluation must run exactly once')
    if tuple(evaluation.get('formal_checkpoint_steps', ())) != CHECKPOINT_STEPS:
        failures.append('evaluation.formal_checkpoint_steps drifted')
    expected_description_schedule = {'core100_steps': [0, 200, 400, 600, 800, 834], 'core100_paired_comparison_from_step0': True, 'final500_steps': [834], 'final500_absolute_report_only': True, 'cross_scope_comparison_forbidden': True}
    if evaluation.get('description_schedule') != expected_description_schedule:
        failures.append('evaluation.description_schedule drifted')
    expected_boundary = {'enabled_steps': [1, 2], 'injection': 'first_microbatch_of_every_rank_local_ga_window', 'max_length': 20000, 'supervised_tokens_min': CERTIFICATION_MIN_SUPERVISED_TOKENS, 'supervised_tokens_max': EXPECTED_MAX_SUPERVISED_TOKENS, 'raw_image_width': CERTIFICATION_IMAGE_WIDTH, 'raw_image_height': CERTIFICATION_IMAGE_HEIGHT, 'processed_pixels': CERTIFICATION_IMAGE_PIXELS, 'disposable_adapter': True}
    if manifest.get('certification', {}).get('boundary_sample') != expected_boundary:
        failures.append('Tier2 worst-case boundary sample contract drifted')
    if tuple(evaluation.get('formal_implementation_files', ())) != FORMAL_EVALUATION_FILES:
        failures.append('evaluation formal implementation file set/order drifted')
    if evaluation.get('python_executable') != software.get('python_executable'):
        failures.append('evaluation must use the certified training Python directly')
    description = evaluation.get('description_current', {})
    expected_description = {'runner': 'scripts/evaluation/evaluate_qwen35b_description_current.py', 'asset_contract': 'scripts/evaluation/qwen35b_description_current_assets.json', 'source_rows': 500, 'source_dataset_sha256': '2c2ca5fc288060257566cecec32a4b207d850ef935c856c13f0601e71fc988e4', 'core100': {'rows': 100, 'source_indices': [0, 1, 2, 4, 8, 13, 19, 23, 26, 38, 41, 53, 58, 59, 64, 69, 73, 89, 90, 97, 110, 113, 120, 122, 123, 130, 144, 145, 157, 160, 166, 167, 171, 172, 174, 180, 182, 188, 189, 191, 201, 208, 213, 214, 219, 221, 223, 227, 230, 231, 238, 243, 248, 251, 253, 260, 263, 281, 288, 298, 300, 301, 302, 310, 311, 313, 316, 326, 340, 346, 354, 359, 369, 371, 375, 380, 385, 391, 393, 399, 402, 406, 411, 412, 426, 434, 448, 464, 466, 467, 469, 471, 473, 477, 484, 485, 487, 494, 496, 499], 'action_bucket_counts': {'click': 20, 'text_key': 20, 'scroll': 20, 'navigation': 20, 'other': 20}, 'seed': 'qwen35b-description-current-stratified100-v1', 'ordered_source_indices_sha256': '72a4cc0b74fb123b7cbe022fc55512ddf30f5ef8af842730a17b4055a23383f9', 'ordered_sample_ids_sha256': '928a8c920c50dd814a366e0edcd0ea6a6cf3680ed794432c26473e85de45aca3', 'ordered_current_image_paths_sha256': 'a0ab042bb82a1584891edd41082f62bd2dc3f13cb5521a5a90977d720d3877d6', 'selection_contract_sha256': '2a53232509e0517cfeb0f3a9eeaec14a842585a386cbaad06b82acf875c1305b', 'materialized_jsonl_sha256': 'ff2eb0544ce29bcad6bd874e4289cac7a3ce3cce71a62d796eee21211a6b1772'}, 'final500': {'rows': 500, 'action_bucket_counts': {'click': 100, 'text_key': 100, 'scroll': 100, 'navigation': 100, 'other': 100}, 'ordered_source_indices_sha256': '5a97a9b5d813c39d121c81830f566f3ef04776fa9ac13eb78a19dc18977041ea', 'absolute_report_only': True}, 'max_new_tokens': 6000, 'max_length': 20000, 'max_output_tokens': 6000, 'device_map': 'balanced', 'cuda_visible_devices': '0,1', 'min_ele_f1_improvement': 0.0, 'max_format_success_drop': 0.01}
    if description != expected_description:
        failures.append('evaluation.description_current drifted')
    action = evaluation.get('action_retention', {})
    expected_action = {'builder': 'scripts/evaluation/build_qwen35b_forgetting_probe.py', 'runner': 'scripts/evaluation/evaluate_qwen35b_forgetting_gate.py', 'bundle_relative_to_output': 'validation/_shared/qwen35b_action_probe', 'production_contract_bundle_path': 'production_contract', 'source_run': 'output/online_mind2web/qwen35b_vanilla_screenshot_bare_stable109_run1_20260728_0015', 'base_model_fingerprint': '009fc1162123d4d03296e122854111949d994127378c450f99b2eeaa665aa419', 'rows': 128, 'device_map': 'balanced', 'cuda_visible_devices': '0,1', 'do_sample': False, 'temperature': 0, 'retries': 0, 'max_new_tokens': 512, 'max_context_tokens': 262144, 'max_memory_gib_each': 76}
    for (name, expected) in expected_action.items():
        if action.get(name) != expected:
            failures.append(f'evaluation.action_retention.{name} drifted')
    expected_recovery = {'public_action': 'recover', 'allowed_source_phases': ['HALTED', 'FAILED', 'EVAL_BLOCKED'], 'require_exact_managed_sessions_absent': True, 'bootstrap_failed_without_preflight_to_new': True, 'bootstrap_requires_no_formal_or_certification_artifacts': True, 'preformal_stale_preflight_config_revision_to_new': True, 'preformal_config_revision_requires_failed_and_no_formal_artifacts': True, 'preformal_config_revision_archives_old_preflight_and_certification': True, 'passed_preflight_requires_current_implementation_match': True, 'certification_failure_quarantines_entire_certification_root': True, 'certification_failure_target_phase': 'PREFLIGHT_PASSED', 'precertification_recovery_auto_launch': False, 'formal_recovery_requires_passed_certification': True, 'formal_recovery_auto_launch': True, 'technical_evaluation_block_recovery_allowed': True, 'failed_quality_comparison_recovery_forbidden': True, 'resume_from_latest_contiguous_passed_prefix': True, 'quality_gate_bypass_allowed': False, 'configuration_change_allowed': False, 'oom_adjustment_requires_new_manifest_and_tier2_recertification': True, 'exact_tmux_watchdog_relaunch': True}
    if manifest.get('recovery') != expected_recovery:
        failures.append('recovery contract drifted')
    if software.get('persistent_tmux') != expected_persistent_tmux:
        failures.append('persistent tmux software contract drifted')
    inventory = toolchain.get('inventory', {}) if isinstance(toolchain, Mapping) else {}
    compiler = toolchain.get('compiler', {}) if isinstance(toolchain, Mapping) else {}
    c_compiler = toolchain.get('c_compiler', {}) if isinstance(toolchain, Mapping) else {}
    critical_files = toolchain.get('critical_files', ()) if isinstance(toolchain, Mapping) else ()
    if toolchain.get('schema_version') != 'guiagent-persistent-cpp-toolchain-v1' or toolchain.get('profile_id') != 'gxx-11.4.0-ubuntu22.04-v1' or toolchain.get('root') != toolchain_root or (toolchain.get('environment') != expected_toolchain_environment) or (compiler.get('path') != compiler_path) or (c_compiler.get('path') != c_compiler_path) or (c_compiler.get('version_line') != 'gcc-11 (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0') or (c_compiler.get('dumpfullversion') != '11.4.0') or (c_compiler.get('dumpmachine') != 'x86_64-linux-gnu') or (compiler.get('dumpfullversion') != '11.4.0') or (compiler.get('dumpmachine') != 'x86_64-linux-gnu'):
        failures.append('persistent C++ toolchain contract drifted')
    expected_inventory = {'algorithm': "sha256(canonical_json(entries)); dir=[path,'dir']; file=[path,'file',sha256,size_bytes,oct(mode&0777)]; symlink=[path,'symlink',raw_target]", 'entry_count': 1157, 'regular_files': 1021, 'symlinks': 59, 'unsafe_symlinks': 0, 'dangling_symlinks': 0, 'total_regular_file_bytes': 167766990, 'sha256': '33a0a6adcdf0e176dcb06fc516bd82fa13973c7e1290cf773a72490a4612e43a'}
    if inventory != expected_inventory:
        failures.append('persistent C++ toolchain inventory contract drifted')
    if not isinstance(critical_files, list) or len(critical_files) != 5:
        failures.append('persistent C++ toolchain critical-file set is incomplete')
    cuda_inventory = cuda_toolchain.get('inventory', {}) if isinstance(cuda_toolchain, Mapping) else {}
    cuda_nvcc = cuda_toolchain.get('nvcc', {}) if isinstance(cuda_toolchain, Mapping) else {}
    cuda_wheels = cuda_toolchain.get('wheel_provenance', ()) if isinstance(cuda_toolchain, Mapping) else ()
    expected_cuda_inventory = {'algorithm': "sha256(canonical_json(entries)); dir=[path,'dir']; file=[path,'file',sha256,size_bytes,oct(mode&0777)]; symlink=[path,'symlink',raw_target]", 'entry_count': 2029, 'regular_files': 1887, 'symlinks': 0, 'unsafe_symlinks': 0, 'dangling_symlinks': 0, 'total_regular_file_bytes': 279154828, 'sha256': '8c2e76b1c4d9cc253de9ab2b471ca9ccd90b6b25057102f90ea66183bddf4f8f'}
    expected_cuda_wheels = {'nvidia-cuda-cccl': ('13.0.50', '46c209499996c18a0c673298f5eac3a09fecdb032da2b8be21c7d0914d966861'), 'nvidia-cuda-crt': ('13.0.88', '2c8043c7c9e02492716426e9919fc78d2c5b3b2a7a768a88e952676b08aa55a4'), 'nvidia-cuda-nvcc': ('13.0.88', '56fe502eb77625a12f25172caa3cdddb4e4c8ba2c8c17dba44b164761b380f03'), 'nvidia-cuda-runtime': ('13.0.96', '7f82250d7782aa23b6cfe765ecc7db554bd3c2870c43f3d1821f1d18aebf0548'), 'nvidia-nvvm': ('13.0.88', 'c5f41ffeb6466944a026dfa5317d7d85355c119bbec279205d22f1869d1054e0')}
    actual_cuda_wheels = {str(item.get('distribution')): (str(item.get('version')), str(item.get('wheel_sha256'))) for item in cuda_wheels if isinstance(item, Mapping)}
    expected_cuda_environment = {'CUDA_HOME': cuda_home, 'PATH_prefix': [f'{cuda_home}/bin'], 'forbid_bundle_in_LD_LIBRARY_PATH': True}
    expected_cuda_smoke_flags = ['--cubin', '-O3', '-lineinfo', '-arch=sm_80', '-std=c++17', '-IMKD2WORLD_EXTERNAL_ROOT/envs/qwen3.5-fast/lib/python3.11/site-packages/tilelang/3rdparty/../src', '-IMKD2WORLD_EXTERNAL_ROOT/envs/qwen3.5-fast/lib/python3.11/site-packages/tilelang/3rdparty/cutlass/include', '--use_fast_math']
    cuda_smoke = cuda_toolchain.get('smoke', {}) if isinstance(cuda_toolchain, Mapping) else {}
    if cuda_toolchain.get('schema_version') != 'guiagent-persistent-cuda-toolchain-v1' or cuda_toolchain.get('profile_id') != 'cuda-13.0.88-cu130-v1' or cuda_toolchain.get('root') != cuda_toolchain_root or (cuda_toolchain.get('cuda_home') != cuda_home) or (cuda_inventory != expected_cuda_inventory) or (actual_cuda_wheels != expected_cuda_wheels) or (cuda_toolchain.get('versions') != {'cuda_version': 13000, 'cudart_version': 13000, 'cccl_version': 3000001}) or (cuda_toolchain.get('environment') != expected_cuda_environment) or (cuda_nvcc.get('path') != nvcc_path) or (cuda_nvcc.get('size_bytes') != 30153384) or (cuda_nvcc.get('version_line') != 'Cuda compilation tools, release 13.0, V13.0.88') or (cuda_nvcc.get('build_line') != 'Build cuda_13.0.r13.0/compiler.36424714_0') or (cuda_nvcc.get('smoke_arch') != 'sm_80') or (cuda_nvcc.get('gpu_execution') is not False) or (cuda_smoke.get('flags') != expected_cuda_smoke_flags) or (cuda_smoke.get('source_includes') != ['tl_templates/cuda/atomic.h', 'cuda/std/__cccl/version.h']) or (cuda_smoke.get('explicit_ccbin') is not False) or (cuda_smoke.get('gpu_execution') is not False):
        failures.append('persistent CUDA/TileLang toolchain contract drifted')
    if jit_cache != expected_jit_cache:
        failures.append('persistent JIT cache contract drifted')
    critical_sha256: dict[str, str] = {'usr/bin/x86_64-linux-gnu-g++-11': str(compiler.get('resolved_sha256', '')), 'usr/bin/x86_64-linux-gnu-gcc-11': str(c_compiler.get('resolved_sha256', ''))}
    for item in critical_files if isinstance(critical_files, list) else ():
        if 'sha256' in item:
            critical_sha256[str(item.get('path', ''))] = str(item['sha256'])
        elif 'resolved_sha256' in item:
            critical_sha256[str(item.get('resolved_path', ''))] = str(item['resolved_sha256'])
    expected_runtime_environment = {'host_toolchain': {'profile_id': toolchain.get('profile_id'), 'root': toolchain.get('root'), 'version': compiler.get('dumpfullversion'), 'inventory_sha256': inventory.get('sha256'), 'critical_sha256': dict(sorted(critical_sha256.items()))}, 'cuda_toolchain': {'profile_id': cuda_toolchain.get('profile_id'), 'root': cuda_toolchain.get('root'), 'cuda_home': cuda_toolchain.get('cuda_home'), 'inventory_sha256': cuda_inventory.get('sha256'), 'nvcc_version': '13.0.88', 'cudart_version_macro': 13000, 'cccl_version_macro': 3000001}, 'nvcc': {'path': cuda_nvcc.get('path'), 'sha256': cuda_nvcc.get('sha256'), 'size_bytes': cuda_nvcc.get('size_bytes'), 'release': '13.0', 'host_compiler': compiler.get('path')}, 'jit_cache_root': jit_cache.get('root') if isinstance(jit_cache, Mapping) else None}
    if manifest.get('runtime_environment') != expected_runtime_environment:
        failures.append('manifest runtime_environment does not bind the resource toolchain')
    if tuple(distributed.get('transformer_layer_cls_to_wrap', ())) != FSDP_WRAP_CLASSES:
        failures.append('FSDP transformer wrap classes drifted')
    if distributed.get('cpu_offload') is not False or distributed.get('cpu_ram_efficient_loading') is not True:
        failures.append('FSDP CPU loading/offload contract drifted')
    if distributed.get('device_map', 'not-null') is not None:
        failures.append('FSDP model load must use device_map=None')
    if activation.get('fsdp_activation_checkpointing') is not False:
        failures.append('Accelerate FSDP activation checkpointing must remain disabled')
    if activation.get('gradient_checkpointing') is not True:
        failures.append('native model gradient checkpointing must remain enabled')
    if activation.get('preserve_rng_state') is not True:
        failures.append('LoRA dropout=0.05 requires preserve_rng_state=True')
    if activation.get('use_reentrant') is not False:
        failures.append('gradient checkpointing must use the non-reentrant implementation')
    if software.get('python_executable') != 'MKD2WORLD_EXTERNAL_ROOT/envs/qwen3.5-fast/bin/python':
        failures.append('training runtime must be locked to the certified qwen3.5-fast Python')
    original_sums_sha = str(model.get('original_sha256sums_sha256', ''))
    if not original_sums.is_absolute():
        failures.append('original model SHA256SUMS path must be absolute and hash-bound')
    if len(original_sums_sha) != 64 or any((character not in '0123456789abcdef' for character in original_sums_sha)):
        failures.append('original model SHA256SUMS SHA256 is invalid')
    for (label, path) in (('canonical model SHA256SUMS', canonical_sums), ('canonicalization receipt', canonical_receipt)):
        if path.is_absolute() or '..' in path.parts or (not path.parts):
            failures.append(f'{label} must use a safe project-relative path')
    if filtering.get('sampling') is not False or filtering.get('stratification') is not False:
        failures.append('current50k must be an exact filter, not a sample')
    if current50k.get('state_side_counts') != {'current': 50000, 'next': 0}:
        failures.append('current50k state-side counts drifted')
    if math.ceil(float(training.get('warmup_ratio', 0.0)) * EXPECTED_MAX_STEPS) != int(training.get('expected_warmup_steps_by_transformers_ceil', -1)):
        failures.append('warmup-step contract is inconsistent with Transformers ceil semantics')
    if failures:
        raise ContractError('manifest/resource contract failed:\n- ' + '\n- '.join(failures))

def build_exposure_plan(*, unique_rows: int=50000, max_steps: int=834, global_batch: int=60) -> dict[str, Any]:
    required = max_steps * global_batch
    if unique_rows > required:
        raise ContractError('unique row count exceeds configured exposure horizon')
    repeats = required - unique_rows
    if repeats != 40:
        raise ContractError(f'expected exactly 40 repeat exposures, found {repeats}')
    ordered_indices = [*range(unique_rows), *range(repeats)]
    return {'schema_version': 'guiagent-current50k-exposure-plan-v1', 'unique_rows': unique_rows, 'repeat_rows': repeats, 'total_exposures': len(ordered_indices), 'ordered_indices': ordered_indices, 'ordered_indices_sha256': ordered_values_sha256(ordered_indices), 'last_step': {'new_indices': ordered_indices[(max_steps - 1) * global_batch:unique_rows], 'repeat_indices': ordered_indices[unique_rows:]}}

def local_exposure_indices(ordered_indices: Sequence[int], *, rank: int, world_size: int, gradient_accumulation_steps: int, start_step: int, end_step: int) -> list[int]:
    if world_size != 2 or gradient_accumulation_steps != 30:
        raise ContractError('rank exposure plan is frozen to world_size=2 and GA=30')
    if rank not in range(world_size):
        raise ContractError(f'invalid rank {rank}')
    if not 0 <= start_step < end_step <= EXPECTED_MAX_STEPS:
        raise ContractError(f'invalid segment [{start_step}, {end_step})')
    global_batch = world_size * gradient_accumulation_steps
    selected: list[int] = []
    for step in range(start_step, end_step):
        block = ordered_indices[step * global_batch:(step + 1) * global_batch]
        if len(block) != global_batch:
            raise ContractError(f'exposure plan is incomplete at step {step + 1}')
        start = rank * gradient_accumulation_steps
        selected.extend(block[start:start + gradient_accumulation_steps])
    return selected

def sampler_state(plan: Mapping[str, Any], *, step: int, sample_ids: Sequence[str], world_size: int=EXPECTED_WORLD_SIZE, gradient_accumulation_steps: int=EXPECTED_GRADIENT_ACCUMULATION) -> dict[str, Any]:
    ordered = plan['ordered_indices']
    global_cursor = step * world_size * gradient_accumulation_steps
    next_indices = ordered[global_cursor:global_cursor + world_size * gradient_accumulation_steps]
    next_ids = [sample_ids[index] for index in next_indices]
    return {'schema_version': 'guiagent-current50k-sampler-state-v1', 'global_step': step, 'global_cursor': global_cursor, 'world_size': world_size, 'gradient_accumulation_steps': gradient_accumulation_steps, 'ordered_exposures_sha256': plan['ordered_indices_sha256'], 'next_indices': next_indices, 'next_sample_ids': next_ids, 'next_sample_ids_sha256': ordered_values_sha256(next_ids)}

def selective_position_lists(labels: Sequence[int], *, ignore_index: int=-100) -> tuple[list[int], list[int]]:
    """Return (logit positions, target positions) for shifted causal CE."""
    target_positions = [index for index in range(1, len(labels)) if labels[index] != ignore_index]
    logit_positions = [index - 1 for index in target_positions]
    if not target_positions:
        raise ContractError('batch has no supervised shifted labels')
    if len(target_positions) > EXPECTED_MAX_SUPERVISED_TOKENS:
        raise ContractError(f'supervised token count {len(target_positions)} exceeds {EXPECTED_MAX_SUPERVISED_TOKENS}')
    return (logit_positions, target_positions)

def token_global_local_scale(local_ce_sum: float, global_num_items: int, world_size: int) -> float:
    if global_num_items <= 0:
        raise ContractError('global supervised-token count must be positive')
    return local_ce_sum / global_num_items * world_size

def validate_transformers_5121_token_scaling_source(get_num_items_source: str, compute_loss_source: str, accelerator_setup_source: str) -> dict[str, bool]:
    denominator_is_global = all((marker in get_num_items_source for marker in ('average_tokens_across_devices', 'self.accelerator.gather', '.sum()')))
    base_applies_world_scale = all((marker in compute_loss_source for marker in ('average_tokens_across_devices', 'loss_scale = self.accelerator.num_processes', 'loss *= loss_scale')))
    accelerate_gas_is_one = 'grad_acc_kwargs["num_steps"] = 1' in accelerator_setup_source or "grad_acc_kwargs['num_steps'] = 1" in accelerator_setup_source
    if not denominator_is_global or not base_applies_world_scale or (not accelerate_gas_is_one):
        raise ContractError('installed Transformers token-scaling source does not match the pinned 5.12.1 contract')
    return {'num_items_in_batch_is_all_rank_sum': True, 'base_compute_loss_applies_world_scale': True, 'accelerate_internal_gradient_accumulation_steps_is_one': True, 'custom_override_must_apply_world_scale_once': True}

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
    selected = sorted(set(selected))
    digest = ordered_values_sha256(selected)
    if len(selected) != EXPECTED_LORA_MODULES:
        raise ContractError(f'Expanded LoRA graph mismatch: modules={len(selected)} sha256={digest}, expected {EXPECTED_LORA_MODULES}/{EXPECTED_TARGET_NAMES_SHA256}')
    return selected

def validate_trainable_adapter(model: Any, contract: AdapterGraphContract=AdapterGraphContract()) -> dict[str, Any]:
    trainable = [(name, parameter) for (name, parameter) in model.named_parameters() if parameter.requires_grad]
    names = [name for (name, _) in trainable]
    non_lora = [name for name in names if 'lora_' not in name]
    parameter_count = sum((int(parameter.numel()) for (_, parameter) in trainable))
    if non_lora:
        raise ContractError(f'non-LoRA parameters are trainable: {non_lora[:10]}')
    if len(trainable) != contract.tensor_count or parameter_count != contract.parameter_count:
        raise ContractError(f'trainable adapter mismatch: tensors={len(trainable)}, parameters={parameter_count}; expected {contract.tensor_count}/{contract.parameter_count}')
    return {'tensor_count': len(trainable), 'parameter_count': parameter_count, 'ordered_parameter_names_sha256': ordered_values_sha256(names)}

def _parse_meta(row: Mapping[str, Any]) -> Mapping[str, Any]:
    meta: Any = row.get('meta')
    if isinstance(meta, str):
        meta = json.loads(meta)
    if not isinstance(meta, Mapping):
        raise ContractError('training row meta must be an object')
    return meta

def sanitize_current_training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    meta = _parse_meta(row)
    if str(meta.get('state_side') or '') != 'current':
        raise ContractError('prepared dataset contains a non-current row')
    if row.get('current_md') != '':
        raise ContractError('current_md must remain the explicit empty string to prevent target leakage')
    target = str(row.get('target_next_md') or '')
    if not target:
        raise ContractError('current row has no target_next_md supervision')
    image_path = row.get('current_image_path') or row.get('image_path')
    if image_path in (None, ''):
        raise ContractError('current row has no current screenshot path')
    identifier = row.get('id') or row.get('sample_id')
    if identifier in (None, ''):
        raise ContractError('current row has no id')
    return {'id': str(identifier), 'task_type': 'localwm_cur', 'current_image_path': str(image_path), 'current_md': '', 'target_next_md': target, 'assistant_output': target, 'action_analysis': '', 'meta': dict(meta), 'meta_json': json.dumps(meta, ensure_ascii=False, sort_keys=True)}

class PlannedCurrentJsonlDataset:
    """Random-access raw JSONL view following an explicit per-rank exposure plan."""

    def __init__(self, path: Path, indices: Sequence[int]) -> None:
        self.path = path
        self.indices = list(indices)
        self.offsets: list[int] = []
        with path.open('rb') as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if len(self.offsets) != 50000:
            raise ContractError(f'prepared current50k has {len(self.offsets)} rows, expected 50000')
        if any((index < 0 or index >= len(self.offsets) for index in self.indices)):
            raise ContractError('exposure plan contains an out-of-range row index')
        self._handle = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        if self._handle is None:
            self._handle = self.path.open('r', encoding='utf-8')
        source_index = self.indices[position]
        self._handle.seek(self.offsets[source_index])
        return sanitize_current_training_row(json.loads(self._handle.readline()))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        value['_handle'] = None
        return value

class CertificationBoundaryDataset:
    """Inject one boundary sample at the start of every rank-local GA window."""

    def __init__(self, base: PlannedCurrentJsonlDataset, boundary_sample: Mapping[str, Any], *, gradient_accumulation_steps: int=EXPECTED_GRADIENT_ACCUMULATION) -> None:
        self.base = base
        self.boundary_sample = dict(boundary_sample)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if len(base) % gradient_accumulation_steps != 0:
            raise ContractError('certification dataset must contain complete GA windows')

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, position: int) -> dict[str, Any]:
        if position % self.gradient_accumulation_steps == 0:
            return dict(self.boundary_sample)
        return self.base[position]

def _tokenizer_length(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)

def build_certification_boundary_target(tokenizer: Any) -> tuple[str, int]:
    """Build deterministic valid-looking current-MD near the 6000-token cap."""
    lines = ['[Environment: Web]']
    token_count = _tokenizer_length(tokenizer, lines[0] + '\n')
    for index in range(10000):
        line = f'[{index}](button): certification boundary control {index} state visible enabled <0,0,999,999>'
        lines.append(line)
        target = '\n'.join(lines) + '\n'
        token_count = _tokenizer_length(tokenizer, target)
        if token_count >= 5940:
            if token_count > 5980:
                raise ContractError(f'certification MD target jumped past the safe near-6000 band: {token_count}')
            return (target, token_count)
    raise ContractError('could not construct the near-6000-token certification MD')

def prepare_certification_boundary_sample(*, processor: Any, image_module: Any, collator: Any, artifact_root: Path, rank: int, torch_module: Any, max_length: int, max_output_tokens: int, max_pixels: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create, collate, and hard-verify the disposable Tier2 boundary sample."""
    if max_length != 20000 or max_output_tokens != 6000 or max_pixels != CERTIFICATION_IMAGE_PIXELS:
        raise ContractError('certification boundary requires frozen 20K/6000/802816 limits')
    artifact_root.mkdir(parents=True, exist_ok=True)
    image_path = artifact_root / 'certification_boundary_896x896.png'
    candidate = artifact_root / f'.certification_boundary_rank{rank}.png'
    image = image_module.new('RGB', (CERTIFICATION_IMAGE_WIDTH, CERTIFICATION_IMAGE_HEIGHT), color=(17, 113, 211))
    image.save(candidate, format='PNG', optimize=False)
    image.close()
    distributed_barrier(torch_module)
    if rank == 0:
        if image_path.is_file():
            candidate.unlink()
        else:
            os.replace(candidate, image_path)
    distributed_barrier(torch_module)
    candidate.unlink(missing_ok=True)
    (target, target_tokens) = build_certification_boundary_target(processor.tokenizer)
    sample = {'id': '__tier2_boundary_current_md__', 'task_type': 'localwm_cur', 'current_image_path': str(image_path), 'current_md': '', 'target_next_md': target, 'assistant_output': target, 'action_analysis': '', 'meta': {'state_side': 'current', 'canonical_domain': 'certification.invalid'}, 'meta_json': json.dumps({'state_side': 'current', 'canonical_domain': 'certification.invalid'}, sort_keys=True)}
    batch = collator([sample])
    labels = batch['labels']
    attention = batch['attention_mask']
    supervised_tokens = int((labels[0, 1:] != -100).sum().item())
    input_tokens = int(attention[0].sum().item())
    pixel_values = batch.get('pixel_values')
    image_grid = batch.get('image_grid_thw')
    if pixel_values is None or image_grid is None or len(image_grid) != 1:
        raise ContractError('certification boundary did not produce image tensors')
    grid = [int(value) for value in image_grid[0].tolist()]
    if len(grid) != 3:
        raise ContractError(f'unexpected image_grid_thw: {grid}')
    patch_size = int(getattr(processor.image_processor, 'patch_size', 14))
    effective_pixels = grid[-2] * grid[-1] * patch_size * patch_size
    if not CERTIFICATION_MIN_SUPERVISED_TOKENS <= supervised_tokens <= EXPECTED_MAX_SUPERVISED_TOKENS:
        raise ContractError(f'certification supervised tokens={supervised_tokens}, expected {CERTIFICATION_MIN_SUPERVISED_TOKENS}..{EXPECTED_MAX_SUPERVISED_TOKENS}')
    if input_tokens > max_length:
        raise ContractError(f'certification input tokens={input_tokens} exceed max_length={max_length}')
    if effective_pixels != max_pixels:
        raise ContractError(f'certification processor pixels={effective_pixels}, expected exact max_pixels={max_pixels}')
    receipt = {'schema_version': 'guiagent-tier2-boundary-sample-v1', 'status': 'passed', 'rank': rank, 'injection_policy': 'first_microbatch_of_every_rank_local_ga_window', 'injected_local_positions': [], 'rank_local_microbatches': 0, 'gradient_accumulation_steps': EXPECTED_GRADIENT_ACCUMULATION, 'ga_windows': 0, 'max_length_config': max_length, 'max_output_tokens_config': max_output_tokens, 'max_pixels_config': max_pixels, 'actual_input_tokens': input_tokens, 'actual_supervised_tokens': supervised_tokens, 'target_tokenizer_tokens': target_tokens, 'raw_image_width': CERTIFICATION_IMAGE_WIDTH, 'raw_image_height': CERTIFICATION_IMAGE_HEIGHT, 'raw_image_pixels': CERTIFICATION_IMAGE_PIXELS, 'image_grid_thw': grid, 'image_patch_size': patch_size, 'effective_processor_pixels': effective_pixels, 'pixel_values_shape': [int(value) for value in pixel_values.shape], 'image_sha256': sha256_file(image_path), 'target_sha256': hashlib.sha256(target.encode('utf-8')).hexdigest()}
    atomic_write_json(artifact_root / f'certification_boundary_rank{rank}.json', receipt)
    del batch
    return (sample, receipt)

def build_training_arguments(TrainingArguments: Any, manifest: Mapping[str, Any], checkpoint_root: Path) -> Any:
    training = manifest['training']
    resource = manifest['_resource_profile']
    distributed = resource['distributed']
    activation = resource['activation_checkpointing']
    kwargs = {'output_dir': str(checkpoint_root), 'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 30, 'max_steps': 834, 'learning_rate': 2e-06, 'lr_scheduler_type': 'cosine', 'warmup_ratio': 0.03, 'weight_decay': 0.01, 'optim': 'adamw_torch_fused', 'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_epsilon': 1e-08, 'max_grad_norm': 1.0, 'seed': 0, 'data_seed': 0, 'bf16': True, 'fp16': False, 'gradient_checkpointing': True, 'gradient_checkpointing_kwargs': {'use_reentrant': activation['use_reentrant'], 'preserve_rng_state': activation['preserve_rng_state']}, 'fsdp': True, 'fsdp_config': {'version': distributed['fsdp_version'], 'reshard_after_forward': distributed['reshard_after_forward'], 'cpu_offload': distributed['cpu_offload'], 'activation_checkpointing': activation['fsdp_activation_checkpointing'], 'cpu_ram_efficient_loading': distributed['cpu_ram_efficient_loading'], 'state_dict_type': distributed['state_dict_type'], 'auto_wrap_policy': distributed['auto_wrap_policy'], 'transformer_layer_cls_to_wrap': distributed['transformer_layer_cls_to_wrap']}, 'save_strategy': 'no', 'save_total_limit': None, 'save_only_model': False, 'logging_steps': training['logging_steps'], 'eval_strategy': 'no', 'remove_unused_columns': False, 'label_names': ['labels'], 'report_to': [], 'dataloader_num_workers': 0, 'dataloader_pin_memory': True, 'dataloader_drop_last': False, 'ignore_data_skip': True, 'average_tokens_across_devices': True, 'disable_tqdm': True}
    signature = inspect.signature(TrainingArguments)
    unsupported = sorted(set(kwargs) - set(signature.parameters))
    if unsupported:
        raise ContractError(f'installed Transformers TrainingArguments lacks required fields: {unsupported}')
    return TrainingArguments(**kwargs)

def build_selective_trainer_class(Trainer: Any, torch_module: Any) -> type:
    functional = torch_module.nn.functional
    DataLoader = torch_module.utils.data.DataLoader

    class SelectiveTokenTrainer(Trainer):

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = True
            if self._loss_shifts_labels is not True:
                raise ContractError('Transformers did not classify Qwen3.5 as shifted causal LM; token denominator would be wrong')

        def get_train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=1, shuffle=False, collate_fn=self.data_collator, num_workers=0, pin_memory=True, drop_last=False)

        def compute_loss(self, model: Any, inputs: Mapping[str, Any], return_outputs: bool=False, num_items_in_batch: Any | None=None):
            model_inputs = dict(inputs)
            labels = model_inputs.pop('labels')
            if labels.ndim != 2 or labels.shape[0] != 1:
                raise ContractError(f'selective CE requires B=1, got labels shape {tuple(labels.shape)}')
            target_positions = torch_module.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten() + 1
            if target_positions.numel() < 1 or target_positions.numel() > EXPECTED_MAX_SUPERVISED_TOKENS:
                raise ContractError(f'supervised shifted tokens={target_positions.numel()}, expected 1..{EXPECTED_MAX_SUPERVISED_TOKENS}')
            logit_positions = target_positions - 1
            outputs = model(**model_inputs, labels=None, logits_to_keep=logit_positions)
            logits = outputs.logits
            if logits.shape[0] != 1 or logits.shape[1] != target_positions.numel():
                raise ContractError(f'selective logits shape {tuple(logits.shape)} does not match {target_positions.numel()} targets')
            targets = labels[0, target_positions].to(logits.device)
            local_sum = functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='sum')
            if not bool(torch_module.isfinite(local_sum).item()):
                raise ContractError('selective CE produced a non-finite local loss sum')
            if num_items_in_batch is None:
                raise ContractError('Trainer did not provide the global GA-window supervised-token count')
            denominator = torch_module.as_tensor(num_items_in_batch, device=local_sum.device, dtype=local_sum.dtype)
            if denominator.item() <= 0:
                raise ContractError('global supervised-token count is not positive')
            loss = local_sum / denominator
            if self.args.average_tokens_across_devices:
                loss = loss * self.accelerator.num_processes
            if not bool(torch_module.isfinite(loss).item()):
                raise ContractError('selective CE produced a non-finite scaled loss')
            return (loss, outputs) if return_outputs else loss
    return SelectiveTokenTrainer

def distributed_barrier(torch_module: Any) -> None:
    if torch_module.distributed.is_available() and torch_module.distributed.is_initialized():
        torch_module.distributed.barrier()

def _adapter_safetensors_summary(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise ContractError('safetensors is required to validate adapter exports') from exc
    tensor_count = 0
    parameter_count = 0
    names: list[str] = []
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            tensor_count += 1
            parameter_count += int(tensor.numel())
            names.append(name)
    if tensor_count != EXPECTED_LORA_TENSORS or parameter_count != EXPECTED_LORA_PARAMETERS:
        raise ContractError(f'standard adapter mismatch: tensors={tensor_count}, parameters={parameter_count}')
    return {'tensor_count': tensor_count, 'parameter_count': parameter_count, 'ordered_tensor_names_sha256': ordered_values_sha256(names), 'sha256': sha256_file(path), 'size_bytes': path.stat().st_size}

def recursive_file_receipt(root: Path, *, excluded: Iterable[str]=()) -> list[dict[str, Any]]:
    excluded_set = set(excluded)
    values: list[dict[str, Any]] = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set:
            continue
        if path.is_symlink():
            raise ContractError(f'checkpoint contains a symlink: {relative}')
        if path.is_file():
            values.append({'path': relative, 'size_bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    return values

def validate_dcp_tree(checkpoint_dir: Path) -> None:
    for directory_name in ('pytorch_model_fsdp_0', 'optimizer_0'):
        directory = checkpoint_dir / directory_name
        if not directory.is_dir():
            raise ContractError(f'missing FSDP DCP directory: {directory_name}')
        if not (directory / '.metadata').is_file():
            raise ContractError(f'missing DCP metadata: {directory_name}/.metadata')
        if not any((path.is_file() and path.name.endswith('.distcp') for path in directory.iterdir())):
            raise ContractError(f'missing DCP shards in {directory_name}')
    forbidden = [path.name for path in checkpoint_dir.iterdir() if path.is_file() and (path.name.startswith('model-') or path.name in {'pytorch_model.bin', 'model.safetensors'})]
    if forbidden:
        raise ContractError(f'full/merged base-model artifacts are forbidden: {forbidden}')

def export_standard_adapter_all_ranks(model: Any, checkpoint_dir: Path, training_args: Any, *, torch_module: Any, rank: int) -> dict[str, Any] | None:
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    except Exception as exc:
        raise ContractError('Torch DCP state_dict APIs required for FSDP2 adapter export') from exc
    state_dict = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True))
    summary = None
    if rank == 0:
        model.save_pretrained(str(checkpoint_dir), state_dict=state_dict, safe_serialization=True)
        torch_module.save(training_args, checkpoint_dir / 'training_args.bin')
        summary = _adapter_safetensors_summary(checkpoint_dir / 'adapter_model.safetensors')
    distributed_barrier(torch_module)
    return summary

def commit_checkpoint(checkpoint_dir: Path, *, step: int, manifest_sha256: str, resource_profile_sha256: str, implementation_sha256s: Mapping[str, str], adapter_summary: Mapping[str, Any], sampler: Mapping[str, Any], memory: Mapping[str, Any], certification_boundary: Mapping[str, Any] | None=None) -> dict[str, Any]:
    validate_dcp_tree(checkpoint_dir)
    if set(memory) != {'rank0', 'rank1'}:
        raise ContractError('checkpoint memory/gradient receipt must contain exactly rank0 and rank1')
    for (rank_name, rank_receipt) in memory.items():
        reserved = int(rank_receipt.get('max_memory_reserved_mib', -1))
        limit = int(rank_receipt.get('maximum_reserved_memory_mib', -1))
        if reserved < 0 or limit <= 0 or reserved > limit:
            raise ContractError(f'{rank_name} memory receipt violates its certified limit')
        if rank_receipt.get('all_trainable_gradients_finite') is not True:
            raise ContractError(f'{rank_name} did not prove finite trainable gradients')
        if int(rank_receipt.get('checked_trainable_gradient_tensors', -1)) != EXPECTED_LORA_TENSORS:
            raise ContractError(f'{rank_name} finite-gradient tensor count drifted')
        if int(rank_receipt.get('finite_gradient_checks', 0)) <= 0:
            raise ContractError(f'{rank_name} has no pre-optimizer finite-gradient check')
    required_files = ['adapter_model.safetensors', 'adapter_config.json', 'scheduler.pt', 'trainer_state.json', 'training_args.bin', 'rng_state_0.pth', 'rng_state_1.pth', 'resume_state_rank0.json', 'resume_state_rank1.json']
    if certification_boundary is not None:
        if set(certification_boundary) != {'rank0', 'rank1'}:
            raise ContractError('certification boundary receipt must contain exactly rank0 and rank1')
        required_files.extend(('certification_boundary_rank0.json', 'certification_boundary_rank1.json'))
    missing = [name for name in required_files if not (checkpoint_dir / name).is_file()]
    if missing:
        raise ContractError(f'checkpoint-{step} is incomplete: missing {missing}')
    trainer_state = load_json(checkpoint_dir / 'trainer_state.json')
    if int(trainer_state.get('global_step', -1)) != step:
        raise ContractError('trainer_state.global_step does not match checkpoint directory')
    resume_state = {f'rank{rank}': load_json(checkpoint_dir / f'resume_state_rank{rank}.json') for rank in range(EXPECTED_WORLD_SIZE)}
    for (rank_name, receipt) in resume_state.items():
        if int(receipt.get('segment_end_step', -1)) != step:
            raise ContractError(f'{rank_name} resume receipt is bound to a different segment end')
        start = int(receipt.get('segment_start_step', -1))
        if int(receipt.get('global_step', -2)) != start:
            raise ContractError(f'{rank_name} resume global step does not match its segment start')
        if start > 0 and (int(receipt.get('optimizer_state_entries', 0)) <= 0 or int(receipt.get('scheduler_last_epoch', -1)) != start):
            raise ContractError(f'{rank_name} did not prove optimizer/scheduler resume')
    atomic_write_json(checkpoint_dir / 'sampler_state.json', sampler)
    files = recursive_file_receipt(checkpoint_dir, excluded=('checkpoint_integrity.json', 'checkpoint_binding.json'))
    integrity = {'schema_version': 'guiagent-fsdp2-checkpoint-integrity-v1', 'status': 'passed', 'global_step': step, 'world_size': EXPECTED_WORLD_SIZE, 'adapter': dict(adapter_summary), 'sampler': dict(sampler), 'memory': dict(memory), 'resume_state': resume_state, 'certification_boundary': dict(certification_boundary or {}), 'files': files, 'files_sha256': canonical_json_sha256(files)}
    integrity_path = checkpoint_dir / 'checkpoint_integrity.json'
    atomic_write_json(integrity_path, integrity)
    binding = {'schema_version': 'guiagent-fsdp2-checkpoint-binding-v1', 'status': 'committed', 'global_step': step, 'world_size': EXPECTED_WORLD_SIZE, 'manifest_sha256': manifest_sha256, 'resource_profile_sha256': resource_profile_sha256, 'implementation_sha256s': dict(sorted(implementation_sha256s.items())), 'integrity_sha256': sha256_file(integrity_path)}
    atomic_write_json(checkpoint_dir / 'checkpoint_binding.json', binding)
    return binding

def finite_trainable_gradient_report(model: Any, *, torch_module: Any) -> dict[str, Any]:
    """Validate every trainable LoRA gradient before the optimizer mutates it.

    FSDP2 exposes sharded parameters/gradients as DTensors. ``to_local`` avoids
    initiating an accidental second DTensor redistribution; the explicit SUM
    below reconstructs the global squared norm across the two shards.
    """
    if not (torch_module.distributed.is_available() and torch_module.distributed.is_initialized() and (torch_module.distributed.get_world_size() == EXPECTED_WORLD_SIZE)):
        raise ContractError('finite-gradient gate requires the certified two-rank process group')
    trainable = [(name, parameter) for (name, parameter) in model.named_parameters() if parameter.requires_grad]
    if len(trainable) != EXPECTED_LORA_TENSORS:
        raise ContractError(f'finite-gradient gate saw {len(trainable)} trainable tensors, expected {EXPECTED_LORA_TENSORS}')
    missing = [name for (name, parameter) in trainable if parameter.grad is None]
    if missing:
        raise ContractError(f'trainable LoRA tensors have no gradient: {missing[:10]}')
    device = torch_module.device('cuda', torch_module.cuda.current_device())
    local_squared_norm = torch_module.zeros((), dtype=torch_module.float64, device=device)
    local_finite = True
    for (_name, parameter) in trainable:
        gradient = parameter.grad.detach()
        to_local = getattr(gradient, 'to_local', None)
        local_gradient = to_local() if callable(to_local) else gradient
        if local_gradient.numel() == 0:
            continue
        local_finite = local_finite and bool(torch_module.isfinite(local_gradient).all().item())
        local_squared_norm += local_gradient.double().square().sum()
    finite_flag = torch_module.tensor(int(local_finite), dtype=torch_module.int32, device=device)
    torch_module.distributed.all_reduce(finite_flag, op=torch_module.distributed.ReduceOp.MIN)
    torch_module.distributed.all_reduce(local_squared_norm, op=torch_module.distributed.ReduceOp.SUM)
    global_norm = float(local_squared_norm.sqrt().item())
    if int(finite_flag.item()) != 1 or not math.isfinite(global_norm):
        raise ContractError(f'non-finite trainable gradient detected; global_norm={global_norm}')
    return {'all_trainable_gradients_finite': True, 'checked_trainable_gradient_tensors': len(trainable), 'missing_gradient_tensors': 0, 'global_gradient_norm_after_clipping': global_norm}

def build_segment_callback(TrainerCallback: Any, *, torch_module: Any, segment_start_step: int, segment_end_step: int, checkpoint_root: Path, manifest_sha256: str, resource_profile_sha256: str, implementation_sha256s: Mapping[str, str], plan: Mapping[str, Any], sample_ids: Sequence[str], rank: int, maximum_reserved_memory_mib: int, certification_boundary_receipt: Mapping[str, Any] | None) -> Any:

    class SegmentCheckpointCallback(TrainerCallback):

        def __init__(self) -> None:
            self.losses: list[float] = []
            self.gradient_reports: list[dict[str, Any]] = []

        def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
            global_step = int(state.global_step)
            if global_step != segment_start_step:
                raise ContractError(f'resume state global_step={global_step}, expected segment start {segment_start_step}')
            optimizer_entries = len(getattr(optimizer, 'state', {})) if optimizer is not None else 0
            if segment_start_step > 0 and optimizer_entries <= 0:
                raise ContractError('resumed optimizer state is empty')
            scheduler_state = lr_scheduler.state_dict() if lr_scheduler is not None else {}
            scheduler_last_epoch = int(scheduler_state.get('last_epoch', -1))
            if segment_start_step > 0 and scheduler_last_epoch != segment_start_step:
                raise ContractError(f'resumed scheduler last_epoch={scheduler_last_epoch}, expected {segment_start_step}')
            receipt = {'schema_version': 'guiagent-fsdp2-resume-state-v1', 'rank': rank, 'global_step': global_step, 'segment_start_step': segment_start_step, 'segment_end_step': segment_end_step, 'optimizer_state_entries': optimizer_entries, 'scheduler_last_epoch': scheduler_last_epoch, 'scheduler_last_lr': list(scheduler_state.get('_last_lr', ()))}
            atomic_write_json(checkpoint_root / f'resume_to_{segment_end_step}_rank{rank}.json', receipt)
            distributed_barrier(torch_module)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                value = float(logs['loss'])
                if not math.isfinite(value):
                    raise ContractError(f'non-finite training loss at step {state.global_step}: {value}')
                self.losses.append(value)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if int(state.global_step) >= segment_end_step:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            report = finite_trainable_gradient_report(model, torch_module=torch_module)
            report['optimizer_step'] = int(state.global_step) + 1
            self.gradient_reports.append(report)
            return control

        def on_save(self, args, state, control, model=None, **kwargs):
            step = int(state.global_step)
            if step != segment_end_step:
                return control
            checkpoint_dir = checkpoint_root / f'checkpoint-{step}'
            adapter_summary = export_standard_adapter_all_ranks(model, checkpoint_dir, args, torch_module=torch_module, rank=rank)
            if not self.gradient_reports or self.gradient_reports[-1]['optimizer_step'] != step:
                raise ContractError(f'checkpoint-{step} has no matching pre-optimizer finite-gradient receipt')
            max_reserved_mib = int(torch_module.cuda.max_memory_reserved() / 1024 ** 2)
            if max_reserved_mib > maximum_reserved_memory_mib:
                raise ContractError(f'rank {rank} max_reserved={max_reserved_mib} MiB exceeds {maximum_reserved_memory_mib} MiB')
            memory = {'rank': rank, 'max_memory_allocated_mib': int(torch_module.cuda.max_memory_allocated() / 1024 ** 2), 'max_memory_reserved_mib': max_reserved_mib, 'maximum_reserved_memory_mib': maximum_reserved_memory_mib, 'finite_logged_losses': len(self.losses), 'finite_gradient_checks': len(self.gradient_reports), **self.gradient_reports[-1]}
            rank_memory = checkpoint_dir / f'memory_rank{rank}.json'
            atomic_write_json(rank_memory, memory)
            resume_receipt = load_json(checkpoint_root / f'resume_to_{segment_end_step}_rank{rank}.json')
            atomic_write_json(checkpoint_dir / f'resume_state_rank{rank}.json', resume_receipt)
            if certification_boundary_receipt is not None:
                atomic_write_json(checkpoint_dir / f'certification_boundary_rank{rank}.json', certification_boundary_receipt)
            distributed_barrier(torch_module)
            if rank == 0:
                combined_memory = {f'rank{item_rank}': load_json(checkpoint_dir / f'memory_rank{item_rank}.json') for item_rank in range(EXPECTED_WORLD_SIZE)}
                combined_boundary = None
                if certification_boundary_receipt is not None:
                    combined_boundary = {f'rank{item_rank}': load_json(checkpoint_dir / f'certification_boundary_rank{item_rank}.json') for item_rank in range(EXPECTED_WORLD_SIZE)}
                commit_checkpoint(checkpoint_dir, step=step, manifest_sha256=manifest_sha256, resource_profile_sha256=resource_profile_sha256, implementation_sha256s=implementation_sha256s, adapter_summary=adapter_summary or {}, sampler=sampler_state(plan, step=step, sample_ids=sample_ids), memory=combined_memory, certification_boundary=combined_boundary)
            distributed_barrier(torch_module)
            control.should_training_stop = True
            return control
    return SegmentCheckpointCallback()

def _load_all_sample_ids(path: Path) -> list[str]:
    values: list[str] = []
    with path.open('r', encoding='utf-8') as handle:
        for (line_number, line) in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get('id') or row.get('sample_id')
            if value in (None, ''):
                raise ContractError(f'prepared row {line_number} has no id')
            values.append(str(value))
    if len(values) != 50000 or len(set(values)) != 50000:
        raise ContractError('prepared current50k IDs are not exactly 50000 unique values')
    return values

def import_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainingArguments
        from mkd2world.training.sft import LocalWMSftCollator
    except Exception as exc:
        raise RuntimeError('training requires torch, transformers, accelerate, peft, Pillow, and the MKD2World package') from exc
    return locals()

def run_segment(args: argparse.Namespace) -> Path:
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    project_root = Path(manifest['identity']['project_root'])
    resource_path = resolve_project_path(project_root, manifest['resource_profile']['path'])
    if not resource_path.is_file():
        resource_path = Path(__file__).resolve().parents[3] / manifest['resource_profile']['path']
    resource = load_json(resource_path)
    validate_manifest_contract(manifest, resource)
    manifest['_resource_profile'] = resource
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size != EXPECTED_WORLD_SIZE or local_rank not in (0, 1):
        raise ContractError(f'trainer requires exactly two ranks, got world_size={world_size}, local_rank={local_rank}')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1':
        raise ContractError('CUDA_VISIBLE_DEVICES must be exactly 0,1')
    if not 0 <= args.segment_start_step < args.segment_end_step <= EXPECTED_MAX_STEPS:
        raise ContractError('invalid segment bounds')
    deps = import_training_dependencies()
    torch = deps['torch']
    validate_transformers_5121_token_scaling_source(inspect.getsource(deps['Trainer']._get_num_items_in_batch), inspect.getsource(deps['Trainer'].compute_loss), inspect.getsource(deps['Trainer'].create_accelerator_and_postprocess))
    torch.cuda.set_device(local_rank)
    checkpoint_root = args.output_root.resolve() / 'checkpoints'
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    training_args = build_training_arguments(deps['TrainingArguments'], manifest, checkpoint_root)
    prepared_path = resolve_project_path(project_root, manifest['data']['current50k']['path'])
    plan_path = resolve_project_path(project_root, manifest['data']['exposure_plan']['path'])
    plan = load_json(plan_path)
    ordered_indices = plan['ordered_indices']
    local_indices = local_exposure_indices(ordered_indices, rank=rank, world_size=world_size, gradient_accumulation_steps=EXPECTED_GRADIENT_ACCUMULATION, start_step=args.segment_start_step, end_step=args.segment_end_step)
    dataset = PlannedCurrentJsonlDataset(prepared_path, local_indices)
    sample_ids = _load_all_sample_ids(prepared_path)
    model_cfg = manifest['model']
    training_cfg = manifest['training']
    processor = deps['AutoProcessor'].from_pretrained(model_cfg['base_model_path'], min_pixels=training_cfg['image_min_pixels'], max_pixels=training_cfg['image_max_pixels'], trust_remote_code=True)
    collator_kwargs = {'processor': processor, 'image_module': deps['Image'], 'max_length': training_cfg['max_length'], 'max_output_tokens': training_cfg['max_output_tokens'], 'assistant_loss_scope': 'full', 'min_supervised_tokens': training_cfg['min_supervised_tokens']}
    collator_signature = inspect.signature(deps['LocalWMSftCollator'])
    if 'prompt_key' not in collator_signature.parameters:
        raise ContractError('installed LocalWMSftCollator lacks the required prompt_key interface')
    collator_kwargs['prompt_key'] = manifest['objective']['prompt_key']
    collator = deps['LocalWMSftCollator'](**collator_kwargs)
    certification_boundary_receipt = None
    if args.certification_boundary:
        if args.segment_end_step - args.segment_start_step != 1:
            raise ContractError('Tier2 boundary certification segments must contain exactly one step')
        (boundary_sample, certification_boundary_receipt) = prepare_certification_boundary_sample(processor=processor, image_module=deps['Image'], collator=collator, artifact_root=args.output_root.resolve() / 'prepared', rank=rank, torch_module=torch, max_length=training_cfg['max_length'], max_output_tokens=training_cfg['max_output_tokens'], max_pixels=training_cfg['image_max_pixels'])
        dataset = CertificationBoundaryDataset(dataset, boundary_sample)
        rank_local_microbatches = len(dataset)
        ga_steps = EXPECTED_GRADIENT_ACCUMULATION
        if rank_local_microbatches % ga_steps != 0:
            raise ContractError('Tier2 boundary dataset does not contain complete GA windows')
        ga_windows = rank_local_microbatches // ga_steps
        injected_positions = list(range(0, rank_local_microbatches, ga_steps))
        if ga_windows < 1 or len(injected_positions) != ga_windows:
            raise ContractError('Tier2 boundary sample does not cover every rank-local GA window')
        certification_boundary_receipt.update({'segment_start_step': args.segment_start_step, 'segment_end_step': args.segment_end_step, 'rank_local_microbatches': rank_local_microbatches, 'gradient_accumulation_steps': ga_steps, 'ga_windows': ga_windows, 'injected_local_positions': injected_positions})
        atomic_write_json(args.output_root.resolve() / 'prepared' / f'certification_boundary_rank{rank}.json', certification_boundary_receipt)
    model = deps['AutoModelForImageTextToText'].from_pretrained(model_cfg['base_model_path'], dtype=torch.bfloat16, device_map=None, attn_implementation=model_cfg['attention_implementation'], trust_remote_code=True)
    model.config.use_cache = False
    model.config.output_router_logits = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    target_modules = expanded_qwen35_lora_target_modules(model)
    lora_cfg = model_cfg['lora']
    model = deps['get_peft_model'](model, deps['LoraConfig'](r=lora_cfg['r'], lora_alpha=lora_cfg['alpha'], lora_dropout=lora_cfg['dropout'], bias='none', task_type='CAUSAL_LM', target_modules=target_modules))
    validate_trainable_adapter(model)
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    implementation_sha256s = {'trainer': sha256_file(Path(__file__).resolve()), **dict(args.implementation_sha256)}
    callback = build_segment_callback(deps['TrainerCallback'], torch_module=torch, segment_start_step=args.segment_start_step, segment_end_step=args.segment_end_step, checkpoint_root=checkpoint_root, manifest_sha256=sha256_file(manifest_path), resource_profile_sha256=sha256_file(resource_path), implementation_sha256s=implementation_sha256s, plan=plan, sample_ids=sample_ids, rank=rank, maximum_reserved_memory_mib=int(manifest['certification']['maximum_reserved_memory_mib_each']), certification_boundary_receipt=certification_boundary_receipt)
    TrainerClass = build_selective_trainer_class(deps['Trainer'], torch)
    trainer = TrainerClass(model=model, args=training_args, train_dataset=dataset, data_collator=collator, callbacks=[callback])
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    if int(trainer.state.global_step) != args.segment_end_step:
        raise ContractError(f'segment exited at step {trainer.state.global_step}, expected {args.segment_end_step}')
    committed = checkpoint_root / f'checkpoint-{args.segment_end_step}' / 'checkpoint_binding.json'
    if not committed.is_file():
        raise ContractError(f'segment ended without committed checkpoint: {committed}')
    return committed

def parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--segment-start-step', type=int, required=True)
    parser.add_argument('--segment-end-step', type=int, required=True)
    parser.add_argument('--resume-from-checkpoint', type=Path)
    parser.add_argument('--certification-boundary', action='store_true', help='Inject and receipt the disposable 20K/6000/max-pixel Tier2 boundary sample.')
    parser.add_argument('--implementation-sha256', action='append', default=[], type=lambda text: tuple(text.split('=', 1)), metavar='NAME=SHA256')
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None=None) -> None:
    args = parse_args(argv)
    committed = run_segment(args)
    if int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps({'status': 'segment_committed', 'binding': str(committed)}, sort_keys=True))
if __name__ == '__main__':
    main()
