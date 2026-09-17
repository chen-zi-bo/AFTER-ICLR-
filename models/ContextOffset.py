import importlib
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.ContextOffsetPrompt import ContextOffsetPromptAssembler, context_json_default
from models.ContextOffsetMemory import (
    EventResponseMemory,
    select_context_state_candidates,
)
from models.ContextOffsetCache import PromptResponseCache



CONTEXT_OFFSET_SETTINGS = {
    'base_model': 'DLinear',
    'llm_backend': 'deepseek',
    'llm_model_name_or_path': './llm/Qwen3-8B',
    'llm_device_map': 'auto',
    'llm_dtype': 'float16',
    'llm_temperature': 0.0,
    'llm_top_p': 1.0,
    'llm_trust_remote_code': 1,
    'llm_local_files_only': 1,
    'llm_load_in_4bit': 0,
    'llm_max_input_tokens': 0,
    'llm_batch_size': 1,
    'stage1_max_new_tokens': 2048,
    'stage2_max_new_tokens': 2048,
    'offset_clip_factor': 0.0,
    'save_prompts': 1,
    'api_base_url': 'https://api.deepseek.com',
    'api_model': 'deepseek-v4-flash',
    'api_key_env': 'DEEPSEEK_API_KEY',
    # Prefer the environment variable named by api_key_env.
    'api_key': '',
    'api_timeout': 120.0,
    'api_max_retries': 3,
    'api_retry_wait': 2.0,
    'api_concurrency': 8,
    'api_response_format': 'json_object',
    'api_thinking_type': 'auto',
    'api_reasoning_effort': 'high',
    'memory_enabled': 0,
    'memory_name': 'continuation_state_memory_v1',
    'memory_dir': './context_memory',
    'memory_path': '',
    'memory_embedding_model_path': './llm/Qwen3-Embedding-0.6B',
    'memory_top_k': 3,
    'memory_rebuild': 0,
    'response_cache_enabled': 1,
    'response_cache_dir': './context_cache',
}

CONTEXT_PROMPT_SETTINGS = {
    'stage1_template_path': './prompt/stage1_context_selection_prompt_template.md',
    'stage2_template_path': './prompt/stage2_context_offset_prompt_template.md',
    'stage2_memory_template_path': './prompt/stage2_context_offset_memory_prompt_template.md',
    'stage1_output_template_path': './prompt/stage1_output.md',
    'table_precision': 4,
    'max_text_chars': 1200,
    'include_empty_texts': 0,
}

_json_default = context_json_default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return float(value)
    except Exception:
        return default


def _has_any_context_text(texts: Any) -> bool:
    if texts is None:
        return False
    if isinstance(texts, str):
        return bool(texts.strip())
    try:
        return any(str(text).strip() for text in texts)
    except TypeError:
        return bool(str(texts).strip())


def _strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith('```'):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith('```'):
        lines = lines[:-1]
    return '\n'.join(lines).strip()


def _remove_leading_plus_from_json_numbers(text: str) -> str:
    chars = []
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if in_string:
            chars.append(char)
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            chars.append(char)
            continue

        if char == '+':
            prev_idx = len(chars) - 1
            while prev_idx >= 0 and chars[prev_idx].isspace():
                prev_idx -= 1
            next_char = text[idx + 1] if idx + 1 < len(text) else ''
            if prev_idx >= 0 and chars[prev_idx] in '[,:' and (next_char.isdigit() or next_char == '.'):
                continue

        chars.append(char)
    return ''.join(chars)


def _extract_json_object(text: str, required_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = _strip_markdown_json_fence(text)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != '{':
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict) and (required_key is None or required_key in obj):
                return obj
        except json.JSONDecodeError:
            continue
    cleaned_text = _remove_leading_plus_from_json_numbers(text)
    if cleaned_text != text:
        for idx, char in enumerate(cleaned_text):
            if char != '{':
                continue
            try:
                obj, _ = decoder.raw_decode(cleaned_text[idx:])
                if isinstance(obj, dict) and (required_key is None or required_key in obj):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def _config_or_default(configs, attr_name: str, default_key: str):
    value = getattr(configs, attr_name, None)
    if value is None:
        return CONTEXT_OFFSET_SETTINGS[default_key]
    return value


def context_memory_path(configs) -> str:
    configured_path = str(_config_or_default(configs, 'context_memory_path', 'memory_path')).strip()
    if configured_path:
        return configured_path
    memory_dir = str(_config_or_default(configs, 'context_memory_dir', 'memory_dir')).strip()
    memory_name = str(_config_or_default(configs, 'context_memory_name', 'memory_name')).strip()
    dataset = str(getattr(configs, 'data', 'dataset')).lower()
    backbone = str(getattr(configs, 'context_base_model', 'backbone'))
    seed = int(getattr(configs, 'seed', 0))
    seq_len = int(getattr(configs, 'seq_len', 0))
    label_len = int(getattr(configs, 'label_len', 0))
    pred_len = int(getattr(configs, 'pred_len', 0))
    file_name = (
        f'{dataset}_{backbone}_s{seed}_sl{seq_len}_ll{label_len}_pl{pred_len}_{memory_name}.jsonl'
    )
    return os.path.join(memory_dir, file_name)


def _normalize_backend(value: Any) -> str:
    backend = str(value or 'hf').strip().lower()
    aliases = {
        'local': 'hf',
        'huggingface': 'hf',
        'api': 'openai_compatible',
        'openai-compatible': 'openai_compatible',
        'openai': 'openai_compatible',
    }
    return aliases.get(backend, backend)


def _resolve_model_path(model_name_or_path: str) -> Tuple[str, bool]:
    value = os.path.expanduser(str(model_name_or_path))
    if os.path.isdir(value):
        return value, True

    if not os.path.isabs(value):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.abspath(os.path.join(project_root, value))
        if os.path.isdir(candidate):
            return candidate, True


    return model_name_or_path, False


class HuggingFaceCausalLLMGenerator:
    def __init__(self, configs):
        self.configs = configs
        configured_model = _config_or_default(configs, 'context_llm_model_name_or_path', 'llm_model_name_or_path')
        self.model_name_or_path, local_path_exists = _resolve_model_path(configured_model)

        self.device_map = _config_or_default(configs, 'context_llm_device_map', 'llm_device_map')
        self.dtype_name = _config_or_default(configs, 'context_llm_dtype', 'llm_dtype')

        self.temperature = float(_config_or_default(configs, 'context_llm_temperature', 'llm_temperature'))
        self.top_p = float(_config_or_default(configs, 'context_llm_top_p', 'llm_top_p'))

        self.trust_remote_code = bool(
            int(_config_or_default(configs, 'context_llm_trust_remote_code', 'llm_trust_remote_code')))
        requested_local_only = bool(
            int(_config_or_default(configs, 'context_llm_local_files_only', 'llm_local_files_only')))
        self.local_files_only = requested_local_only or local_path_exists
        self.load_in_4bit = bool(int(_config_or_default(configs, 'context_llm_load_in_4bit', 'llm_load_in_4bit')))
        # 输入无上限
        self.max_input_tokens = int(_config_or_default(configs, 'context_llm_max_input_tokens', 'llm_max_input_tokens'))
        self.batch_size = max(1, int(
            _config_or_default(configs, 'context_llm_batch_size', 'llm_batch_size')
        ))
        self.tokenizer = None
        self.model = None

    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: Sequence[str], **kwargs) -> List[str]:
        self._ensure_loaded()
        responses = []
        max_new_tokens = int(kwargs.get('max_new_tokens', 512))
        for start in range(0, len(prompts), self.batch_size):
            chunk = list(prompts[start:start + self.batch_size])
            responses.extend(self._generate_prompt_chunk(chunk, max_new_tokens))
        return responses

    def _generate_prompt_chunk(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        texts = []
        for prompt in prompts:
            messages = [{'role': 'user', 'content': prompt}]
            if hasattr(self.tokenizer, 'apply_chat_template'):
                texts.append(self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ))
            else:
                texts.append(prompt)

        token_kwargs = {
            'return_tensors': 'pt',
            'padding': len(texts) > 1,
            'truncation': self.max_input_tokens > 0,
        }
        if self.max_input_tokens > 0:
            token_kwargs['max_length'] = self.max_input_tokens
        inputs = self.tokenizer(texts, **token_kwargs)
        device = self._input_device()
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': self.temperature > 0,
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs['temperature'] = self.temperature
            generation_kwargs['top_p'] = self.top_p

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation_kwargs)
        except RuntimeError as exc:
            if len(prompts) <= 1 or 'out of memory' not in str(exc).lower():
                raise
            del inputs
            torch.cuda.empty_cache()
            midpoint = len(prompts) // 2
            return (
                self._generate_prompt_chunk(prompts[:midpoint], max_new_tokens)
                + self._generate_prompt_chunk(prompts[midpoint:], max_new_tokens)
            )
        prompt_width = inputs['input_ids'].shape[-1]
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(
                output_ids[:, prompt_width:],
                skip_special_tokens=True,
            )
        ]

    def _ensure_loaded(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = self._torch_dtype()
        model_kwargs = {
            'torch_dtype': dtype,
            'device_map': self.device_map,
            'trust_remote_code': self.trust_remote_code,
            'local_files_only': self.local_files_only,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)

        print(
            f'[ContextOffset] Loading LLM from {self.model_name_or_path} '
            f'(local_files_only={self.local_files_only})'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kwargs,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.model.eval()

    def _input_device(self) -> torch.device:
        if self.model is None:
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        try:
            embeddings = self.model.get_input_embeddings()
            if embeddings is not None and hasattr(embeddings, 'weight'):
                return embeddings.weight.device
        except Exception:
            pass

        device_map = getattr(self.model, 'hf_device_map', None)
        if isinstance(device_map, dict):
            for target in device_map.values():
                if isinstance(target, str) and target.startswith('cuda'):
                    return torch.device(target)
                if isinstance(target, int):
                    return torch.device(f'cuda:{target}')

        return getattr(self.model, 'device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def _torch_dtype(self):
        dtype_name = str(self.dtype_name).lower()
        if dtype_name in {'bf16', 'bfloat16'}:
            return torch.bfloat16
        if dtype_name in {'fp32', 'float32'}:
            return torch.float32
        return torch.float16


class OpenAICompatibleChatGenerator:
    def __init__(self, configs, backend: str):
        self.configs = configs
        self.backend = backend
        self.base_url = str(_config_or_default(configs, 'context_api_base_url', 'api_base_url')).strip()
        self.model = str(_config_or_default(configs, 'context_api_model', 'api_model')).strip()
        self.api_key_env = str(_config_or_default(configs, 'context_api_key_env', 'api_key_env')).strip()
        self.api_key = _config_or_default(configs, 'context_api_key', 'api_key')
        if self.api_key is None or str(self.api_key).strip() == '':
            self.api_key = CONTEXT_OFFSET_SETTINGS.get('api_key', '')
        if self.api_key is None or str(self.api_key).strip() == '':
            self.api_key = os.environ.get(self.api_key_env, '')
        self.api_key = str(self.api_key).strip()
        self.timeout = float(_config_or_default(configs, 'context_api_timeout', 'api_timeout'))
        self.max_retries = int(_config_or_default(configs, 'context_api_max_retries', 'api_max_retries'))
        self.retry_wait = float(_config_or_default(configs, 'context_api_retry_wait', 'api_retry_wait'))
        self.temperature = float(_config_or_default(configs, 'context_llm_temperature', 'llm_temperature'))
        self.top_p = float(_config_or_default(configs, 'context_llm_top_p', 'llm_top_p'))
        self.top_k = getattr(configs, 'context_api_top_k', None)
        self.min_p = getattr(configs, 'context_api_min_p', None)
        enable_thinking = getattr(configs, 'context_api_enable_thinking', None)
        self.enable_thinking = (
            None if enable_thinking is None else bool(int(enable_thinking))
        )
        self.response_format = str(
            _config_or_default(configs, 'context_api_response_format', 'api_response_format')
        ).strip().lower()
        self.thinking_type = str(
            _config_or_default(configs, 'context_api_thinking_type', 'api_thinking_type')
        ).strip().lower()
        self.reasoning_effort = str(
            _config_or_default(configs, 'context_api_reasoning_effort', 'api_reasoning_effort')
        ).strip()

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            raise RuntimeError(
                f'AFTER API key is missing. Set the {self.api_key_env} environment variable '
                'or pass --context_api_key.'
            )

        payload = self._payload(prompt, int(kwargs.get('max_new_tokens', 512)))
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        url = self._chat_completions_url()
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }

        last_error = None
        attempts = max(1, self.max_retries + 1)
        for attempt in range(attempts):
            req = urllib.request.Request(url, data=body, headers=headers, method='POST')
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    response_body = response.read().decode('utf-8')
                data = json.loads(response_body)
                return self._extract_content(data)
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode('utf-8', errors='replace')
                last_error = RuntimeError(f'HTTP {exc.code} from {url}: {error_body}')
                if exc.code < 500 and exc.code not in {408, 409, 429}:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc

            if attempt + 1 < attempts:
                time.sleep(self.retry_wait * (attempt + 1))

        raise RuntimeError(f'ContextOffset API generation failed: {last_error}') from last_error

    def _payload(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': max_tokens,
            'stream': False,
        }
        if self.response_format == 'json_object':
            payload['response_format'] = {'type': 'json_object'}

        # Extra sampling controls are supported only by compatible backends.
        # Keep them out of DeepSeek requests so its API behavior is unchanged.
        if self.backend == 'openai_compatible':
            if self.top_k is not None:
                payload['top_k'] = int(self.top_k)
            if self.min_p is not None:
                payload['min_p'] = float(self.min_p)
            if self.enable_thinking is not None:
                payload['chat_template_kwargs'] = {
                    'enable_thinking': self.enable_thinking,
                }

        thinking_type = self.thinking_type
        if thinking_type == 'auto':
            thinking_type = 'disabled' if self.backend == 'deepseek' else 'none'
        if thinking_type != 'none':
            payload['thinking'] = {'type': thinking_type}
            if thinking_type == 'enabled' and self.reasoning_effort:
                payload['reasoning_effort'] = self.reasoning_effort
        return payload

    def _chat_completions_url(self) -> str:
        base_url = self.base_url.rstrip('/')
        if base_url.endswith('/chat/completions'):
            return base_url
        return f'{base_url}/chat/completions'

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        choices = data.get('choices')
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f'API response has no choices: {data}')
        message = choices[0].get('message', {})
        content = message.get('content', '')
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get('text', item.get('content', ''))))
                else:
                    parts.append(str(item))
            return ''.join(parts).strip()
        return str(content).strip()


def build_context_llm_generator(configs):
    backend = _normalize_backend(_config_or_default(configs, 'context_llm_backend', 'llm_backend'))
    if backend == 'hf':
        return HuggingFaceCausalLLMGenerator(configs)
    if backend in {'deepseek', 'openai_compatible'}:
        return OpenAICompatibleChatGenerator(configs, backend=backend)
    raise ValueError(f'Unknown ContextOffset LLM backend: {backend}')


class ContextOffsetReasoner:
    def __init__(self, configs, load_memory: bool = True):
        self.configs = configs
        self.prompt_builder = ContextOffsetPromptAssembler(configs, CONTEXT_PROMPT_SETTINGS)
        self.llm = build_context_llm_generator(configs)
        self.stage1_max_new_tokens = int(
            _config_or_default(configs, 'context_stage1_max_new_tokens', 'stage1_max_new_tokens'))
        self.stage2_max_new_tokens = int(
            _config_or_default(configs, 'context_stage2_max_new_tokens', 'stage2_max_new_tokens'))
        # 约束偏移量
        self.clip_factor = float(_config_or_default(configs, 'context_offset_clip_factor', 'offset_clip_factor'))
        self.backend = _normalize_backend(_config_or_default(configs, 'context_llm_backend', 'llm_backend'))
        api_concurrency = int(_config_or_default(configs, 'context_api_concurrency', 'api_concurrency'))
        self.sample_concurrency = api_concurrency if self.backend in {'deepseek', 'openai_compatible'} else 1
        self.sample_concurrency = max(1, self.sample_concurrency)
        self.memory_enabled = bool(int(_config_or_default(configs, 'context_memory_enabled', 'memory_enabled')))
        self.memory = self._load_memory() if self.memory_enabled and load_memory else None
        cache_enabled = bool(int(_config_or_default(
            configs, 'context_response_cache_enabled', 'response_cache_enabled')))
        cache_dir = str(_config_or_default(
            configs, 'context_response_cache_dir', 'response_cache_dir')).strip()
        model_name = (
            _config_or_default(configs, 'context_api_model', 'api_model')
            if self.backend in {'deepseek', 'openai_compatible'}
            else _config_or_default(configs, 'context_llm_model_name_or_path', 'llm_model_name_or_path')
        )
        self.response_cache = PromptResponseCache(
            cache_dir=cache_dir,
            namespace={
                'backend': self.backend,
                'model': str(model_name),
                'temperature': float(_config_or_default(
                    configs, 'context_llm_temperature', 'llm_temperature')),
                'top_p': float(_config_or_default(configs, 'context_llm_top_p', 'llm_top_p')),
            },
            enabled=cache_enabled,
        )

    def _memory_path(self) -> str:
        return context_memory_path(self.configs)

    def _generate(self, stage: str, prompt: str, max_new_tokens: int) -> Tuple[str, bool]:
        return self.response_cache.get_or_generate(
            stage=stage,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            generate=lambda: self.llm.generate(prompt, max_new_tokens=max_new_tokens),
        )

    def _generate_batch(
            self,
            stage: str,
            prompts: Sequence[str],
            max_new_tokens: int,
    ) -> List[Tuple[str, bool]]:
        outputs: List[Optional[Tuple[str, bool]]] = [None] * len(prompts)
        missing_indices = []
        missing_prompts = []
        for idx, prompt in enumerate(prompts):
            cached, _ = self.response_cache.lookup(stage, prompt, max_new_tokens)
            if cached is None:
                missing_indices.append(idx)
                missing_prompts.append(prompt)
            else:
                outputs[idx] = (cached, True)

        if missing_prompts:
            if hasattr(self.llm, 'generate_batch'):
                llm_batch_size = max(1, int(getattr(self.llm, 'batch_size', 1)))
                generated_batches = []
                for start in range(0, len(missing_prompts), llm_batch_size):
                    prompt_batch = missing_prompts[start:start + llm_batch_size]
                    response_batch = self.llm.generate_batch(
                        prompt_batch,
                        max_new_tokens=max_new_tokens,
                    )
                    if len(response_batch) != len(prompt_batch):
                        raise RuntimeError(
                            'Batch LLM generation returned an unexpected response count.'
                        )
                    for idx, prompt, response in zip(
                            missing_indices[start:start + llm_batch_size],
                            prompt_batch,
                            response_batch,
                    ):
                        outputs[idx] = self.response_cache.get_or_generate(
                            stage=stage,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            generate=lambda response=response: response,
                        )
                    generated_batches.extend(response_batch)
                responses = generated_batches
            else:
                responses = [
                    self.llm.generate(prompt, max_new_tokens=max_new_tokens)
                    for prompt in missing_prompts
                ]
            if len(responses) != len(missing_prompts):
                raise RuntimeError('Batch LLM generation returned an unexpected response count.')
            for idx, prompt, response in zip(missing_indices, missing_prompts, responses):
                if outputs[idx] is not None:
                    continue
                outputs[idx] = self.response_cache.get_or_generate(
                    stage=stage,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    generate=lambda response=response: response,
                )
        return [output for output in outputs if output is not None]

    def _load_memory(self) -> EventResponseMemory:
        path = self._memory_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'ContextOffset memory is enabled but no memory file exists at {path}. '
                'Build the memory once before memory-enabled test inference.'
            )
        return EventResponseMemory.load(
            path,
            embedding_model_path=str(_config_or_default(
                self.configs, 'context_memory_embedding_model_path', 'memory_embedding_model_path')),
            top_k=int(_config_or_default(
                self.configs, 'context_memory_top_k', 'memory_top_k')),
        )

    def run_batch(
            self,
            history_values: np.ndarray,
            history_times: Sequence[Sequence[str]],
            context_texts: Sequence[Sequence[str]],
            future_times: Sequence[Sequence[str]],
            base_forecast: np.ndarray,
            variables: Sequence[str],
            sample_indices: Optional[Sequence[Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:

        base_forecast = np.asarray(base_forecast, dtype=float)
        variables = self.prompt_builder._normalize_variables(variables, base_forecast.shape[-1])
        offsets = np.zeros_like(base_forecast, dtype=float)
        batch_size = base_forecast.shape[0]

        if self.backend == 'hf' and getattr(self.llm, 'batch_size', 1) > 1:
            return self._run_hf_batch(
                history_values=history_values,
                history_times=history_times,
                context_texts=context_texts,
                future_times=future_times,
                base_forecast=base_forecast,
                variables=variables,
                sample_indices=sample_indices,
            )

        if self.sample_concurrency <= 1 or batch_size <= 1:
            records: List[Dict[str, Any]] = []
            for row_idx in range(batch_size):
                row_idx, sample_offsets, record = self._run_one_sample(
                    row_idx=row_idx,
                    history_values=history_values,
                    history_times=history_times,
                    context_texts=context_texts,
                    future_times=future_times,
                    base_forecast=base_forecast,
                    variables=variables,
                    sample_indices=sample_indices,
                )
                offsets[row_idx] = sample_offsets
                records.append(record)
            return base_forecast + offsets, offsets, records

        records_by_row: List[Optional[Dict[str, Any]]] = [None] * batch_size
        worker_count = min(self.sample_concurrency, batch_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    self._run_one_sample,
                    row_idx=row_idx,
                    history_values=history_values,
                    history_times=history_times,
                    context_texts=context_texts,
                    future_times=future_times,
                    base_forecast=base_forecast,
                    variables=variables,
                    sample_indices=sample_indices,
                )
                for row_idx in range(batch_size)
            ]
            for future in concurrent.futures.as_completed(futures):
                row_idx, sample_offsets, record = future.result()
                offsets[row_idx] = sample_offsets
                records_by_row[row_idx] = record

        records = [record for record in records_by_row if record is not None]
        return base_forecast + offsets, offsets, records

    def _run_hf_batch(
            self,
            history_values: np.ndarray,
            history_times: Sequence[Sequence[str]],
            context_texts: Sequence[Sequence[str]],
            future_times: Sequence[Sequence[str]],
            base_forecast: np.ndarray,
            variables: Sequence[str],
            sample_indices: Optional[Sequence[Any]],
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        batch_started = time.perf_counter()
        batch_size = len(base_forecast)
        stage1_results = self.run_stage1_batch(
            history_values=history_values,
            history_times=history_times,
            context_texts=context_texts,
            future_times=future_times,
            variables=variables,
        )

        stage2_prompts = []
        for row_idx, stage1_result in enumerate(stage1_results):
            stage2_prompts.append(self.prompt_builder.build_stage2_prompt(
                history_values=np.asarray(history_values[row_idx], dtype=float),
                history_times=list(history_times[row_idx]),
                future_times=list(future_times[row_idx]),
                base_forecast=base_forecast[row_idx],
                variables=variables,
                stage1_json=stage1_result['stage1_json'],
            ))
        stage2_started = time.perf_counter()
        stage2_generated = self._generate_batch(
            'stage2', stage2_prompts, self.stage2_max_new_tokens
        )
        stage2_seconds = (time.perf_counter() - stage2_started) / max(1, batch_size)

        no_memory_jsons = []
        no_memory_offsets = []
        memory_retrievals = []
        memory_prompts: List[Optional[str]] = [None] * batch_size
        memory_positions = []
        for row_idx, ((stage2_response, _), stage1_result) in enumerate(
                zip(stage2_generated, stage1_results)):
            parsed = _extract_json_object(
                stage2_response, required_key='context_offset_values'
            ) or {}
            no_memory_jsons.append(parsed)
            no_memory_offsets.append(self._parse_offsets(
                parsed, list(future_times[row_idx]), variables
            ))

            retrieval = None
            examples = []
            if self.memory is not None:
                retrieval = self.memory.retrieve(
                    history_values=np.asarray(history_values[row_idx], dtype=float),
                    base_forecast=base_forecast[row_idx],
                    stage1_json=stage1_result['stage1_json'],
                )
                examples = retrieval.get('examples', [])
            memory_retrievals.append(retrieval)
            if examples:
                memory_prompts[row_idx] = self.prompt_builder.build_stage2_memory_prompt(
                    history_values=np.asarray(history_values[row_idx], dtype=float),
                    history_times=list(history_times[row_idx]),
                    future_times=list(future_times[row_idx]),
                    base_forecast=base_forecast[row_idx],
                    variables=variables,
                    stage1_json=stage1_result['stage1_json'],
                    memory_examples=examples,
                )
                memory_positions.append(row_idx)

        memory_generated_by_row = {}
        memory_seconds = 0.0
        if memory_positions:
            memory_started = time.perf_counter()
            generated = self._generate_batch(
                'stage2_memory',
                [memory_prompts[row_idx] for row_idx in memory_positions],
                self.stage2_max_new_tokens,
            )
            memory_seconds = (
                time.perf_counter() - memory_started
            ) / max(1, len(memory_positions))
            memory_generated_by_row = dict(zip(memory_positions, generated))

        offsets = np.zeros_like(base_forecast, dtype=float)
        records = []
        sample_seconds = (time.perf_counter() - batch_started) / max(1, batch_size)
        save_prompts = bool(int(_config_or_default(
            self.configs, 'context_save_prompts', 'save_prompts'
        )))
        for row_idx, stage1_result in enumerate(stage1_results):
            stage2_response, stage2_cache_hit = stage2_generated[row_idx]
            sample_offsets = no_memory_offsets[row_idx]
            memory_prompt = memory_prompts[row_idx]
            memory_response = None
            memory_json = None
            memory_cache_hit = False
            fusion = {
                'used_memory': False,
                'method': 'context_only',
                'reason': 'no_reliable_memory_examples',
                'context_only_offset': sample_offsets,
                'final_offset': sample_offsets,
            }
            if row_idx in memory_generated_by_row:
                memory_response, memory_cache_hit = memory_generated_by_row[row_idx]
                memory_json = _extract_json_object(
                    memory_response, required_key='context_offset_values'
                )
                if memory_json is not None:
                    parsed_memory_offsets = self._parse_offsets(
                        memory_json, list(future_times[row_idx]), variables
                    )
                    sample_offsets, fusion = self._fuse_offsets(
                        no_memory_offsets=no_memory_offsets[row_idx],
                        memory_offsets=parsed_memory_offsets,
                        memory_retrieval=memory_retrievals[row_idx],
                    )
                else:
                    fusion = {
                        'used_memory': False,
                        'method': 'context_only',
                        'reason': 'memory_stage2_parse_failed',
                        'context_only_offset': sample_offsets,
                        'final_offset': sample_offsets,
                    }

            sample_offsets = self._clip_offsets(
                sample_offsets,
                np.asarray(history_values[row_idx], dtype=float),
                variables,
            )
            offsets[row_idx] = sample_offsets
            fusion['final_offset'] = sample_offsets
            idx_value = (
                None if sample_indices is None
                else _json_default(sample_indices[row_idx])
            )
            record = {
                'sample_index': idx_value,
                'history_times': list(history_times[row_idx]),
                'future_times': list(future_times[row_idx]),
                'stage1_json': stage1_result['stage1_json'],
                'stage1_parse_ok': stage1_result['parse_ok'],
                'stage1_cache_hit': stage1_result['cache_hit'],
                'stage2_nomemory_cache_hit': stage2_cache_hit,
                'stage1_response': stage1_result['response'],
                'stage2_nomemory_json': no_memory_jsons[row_idx],
                'stage2_nomemory_response': stage2_response,
                'stage2_memory_json': memory_json,
                'stage2_memory_response': memory_response,
                'stage2_memory_cache_hit': memory_cache_hit,
                'fusion': fusion,
                'no_memory_context_offset': no_memory_offsets[row_idx],
                'context_offset': sample_offsets,
                'timing_seconds': {
                    'stage1': stage1_result['timing_seconds'],
                    'stage2_nomemory': stage2_seconds,
                    'stage2_memory': memory_seconds if memory_prompt else 0.0,
                    'sample_total': sample_seconds,
                },
            }
            if memory_retrievals[row_idx] is not None:
                record['memory_retrieval'] = self._compact_memory_retrieval(
                    memory_retrievals[row_idx]
                )
            if save_prompts:
                record['stage1_prompt'] = stage1_result['prompt']
                record['stage2_nomemory_prompt'] = stage2_prompts[row_idx]
                record['stage2_memory_prompt'] = memory_prompt
            records.append(record)
        return base_forecast + offsets, offsets, records

    def run_stage1_batch(
            self,
            history_values: np.ndarray,
            history_times: Sequence[Sequence[str]],
            context_texts: Sequence[Sequence[str]],
            future_times: Sequence[Sequence[str]],
            variables: Sequence[str],
    ) -> List[Dict[str, Any]]:
        batch_size = len(history_values)
        outputs: List[Optional[Dict[str, Any]]] = [None] * batch_size

        if self.backend == 'hf' and getattr(self.llm, 'batch_size', 1) > 1:
            prompts = [
                self.prompt_builder.build_stage1_prompt(
                    history_values=np.asarray(history_values[row_idx], dtype=float),
                    history_times=list(history_times[row_idx]),
                    future_times=list(future_times[row_idx]),
                    context_texts=list(context_texts[row_idx]),
                    variables=variables,
                )
                for row_idx in range(batch_size)
            ]
            started = time.perf_counter()
            generated = self._generate_batch(
                'stage1', prompts, self.stage1_max_new_tokens
            )
            elapsed_per_sample = (time.perf_counter() - started) / max(1, batch_size)
            for row_idx, (response, cache_hit) in enumerate(generated):
                parsed = _extract_json_object(response, required_key='context_status')
                outputs[row_idx] = {
                    'prompt': prompts[row_idx],
                    'response': response,
                    'stage1_json': parsed or {
                        'context_status': 'none',
                        'history_state_summary': '',
                        'context_summary': 'No parseable forecast-relevant context state was returned.',
                        'evidence_times': [],
                    },
                    'parse_ok': parsed is not None,
                    'cache_hit': cache_hit,
                    'timing_seconds': elapsed_per_sample,
                }
            return [result for result in outputs if result is not None]

        def run_one(row_idx: int) -> Tuple[int, Dict[str, Any]]:
            result = self._select_context(
                history_values=np.asarray(history_values[row_idx], dtype=float),
                history_times=list(history_times[row_idx]),
                future_times=list(future_times[row_idx]),
                context_texts=list(context_texts[row_idx]),
                variables=variables,
            )
            return row_idx, result

        if self.sample_concurrency == 1 or batch_size <= 1:
            for row_idx in range(batch_size):
                idx, result = run_one(row_idx)
                outputs[idx] = result
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.sample_concurrency) as executor:
                futures = [executor.submit(run_one, row_idx) for row_idx in range(batch_size)]
                for future in concurrent.futures.as_completed(futures):
                    idx, result = future.result()
                    outputs[idx] = result
        return [result for result in outputs if result is not None]

    def _select_context(
            self,
            history_values: np.ndarray,
            history_times: Sequence[str],
            future_times: Sequence[str],
            context_texts: Sequence[str],
            variables: Sequence[str],
    ) -> Dict[str, Any]:
        prompt = self.prompt_builder.build_stage1_prompt(
            history_values=history_values,
            history_times=history_times,
            future_times=future_times,
            context_texts=context_texts,
            variables=variables,
        )
        started = time.perf_counter()
        response, cache_hit = self._generate('stage1', prompt, self.stage1_max_new_tokens)
        elapsed = time.perf_counter() - started
        parsed = _extract_json_object(response, required_key='context_status')
        empty_context_state = {
            'context_status': 'none',
            'history_state_summary': '',
            'context_summary': 'No parseable forecast-relevant context state was returned.',
            'evidence_times': [],
        }
        return {
            'prompt': prompt,
            'response': response,
            'stage1_json': parsed or empty_context_state,
            'parse_ok': parsed is not None,
            'cache_hit': cache_hit,
            'timing_seconds': elapsed,
        }

    def _run_one_sample(
            self,
            row_idx: int,
            history_values: np.ndarray,
            history_times: Sequence[Sequence[str]],
            context_texts: Sequence[Sequence[str]],
            future_times: Sequence[Sequence[str]],
            base_forecast: np.ndarray,
            variables: Sequence[str],
            sample_indices: Optional[Sequence[Any]],
    ) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        sample_start_time = time.perf_counter()
        idx_value = None if sample_indices is None else _json_default(sample_indices[row_idx])

        hist = np.asarray(history_values[row_idx], dtype=float)
        hist_times = list(history_times[row_idx])
        fut_times = list(future_times[row_idx])
        texts = list(context_texts[row_idx])

        stage1_result = self._select_context(
            history_values=hist,
            history_times=hist_times,
            future_times=fut_times,
            context_texts=texts,
            variables=variables,
        )
        stage1_prompt = stage1_result['prompt']
        stage1_response = stage1_result['response']
        stage1_seconds = stage1_result['timing_seconds']
        stage1_parse_ok = stage1_result['parse_ok']
        stage1_json = stage1_result['stage1_json']

        memory_retrieval = None
        memory_examples = []
        if self.memory is not None:
            memory_retrieval = self.memory.retrieve(
                history_values=hist,
                base_forecast=base_forecast[row_idx],
                stage1_json=stage1_json,
            )
            memory_examples = memory_retrieval.get('examples', [])

        stage2_prompt = self.prompt_builder.build_stage2_prompt(
            history_values=hist,
            history_times=hist_times,
            future_times=fut_times,
            base_forecast=base_forecast[row_idx],
            variables=variables,
            stage1_json=stage1_json,
        )
        stage2_start_time = time.perf_counter()
        stage2_response, stage2_cache_hit = self._generate(
            'stage2', stage2_prompt, self.stage2_max_new_tokens)
        stage2_seconds = time.perf_counter() - stage2_start_time
        no_memory_json = _extract_json_object(
            stage2_response, required_key='context_offset_values') or {}
        no_memory_offsets = self._parse_offsets(no_memory_json, fut_times, variables)

        memory_prompt = None
        memory_response = None
        memory_json = None
        memory_cache_hit = False
        memory_seconds = 0.0
        sample_offsets = no_memory_offsets
        fusion = {
            'used_memory': False,
            'method': 'context_only',
            'reason': 'no_reliable_memory_examples',
            'context_only_offset': no_memory_offsets,
            'final_offset': no_memory_offsets,
        }
        if memory_examples:
            memory_prompt = self.prompt_builder.build_stage2_memory_prompt(
                history_values=hist,
                history_times=hist_times,
                future_times=fut_times,
                base_forecast=base_forecast[row_idx],
                variables=variables,
                stage1_json=stage1_json,
                memory_examples=memory_examples,
            )
            memory_start_time = time.perf_counter()
            memory_response, memory_cache_hit = self._generate(
                'stage2_memory', memory_prompt, self.stage2_max_new_tokens)
            memory_seconds = time.perf_counter() - memory_start_time
            memory_json = _extract_json_object(
                memory_response, required_key='context_offset_values')
            if memory_json is not None:
                memory_offsets = self._parse_offsets(memory_json, fut_times, variables)
                sample_offsets, fusion = self._fuse_offsets(
                    no_memory_offsets=no_memory_offsets,
                    memory_offsets=memory_offsets,
                    memory_retrieval=memory_retrieval,
                )
            else:
                fusion = {
                    'used_memory': False,
                    'method': 'context_only',
                    'reason': 'memory_stage2_parse_failed',
                    'context_only_offset': no_memory_offsets,
                    'final_offset': no_memory_offsets,
                }

        sample_offsets = self._clip_offsets(sample_offsets, hist, variables)
        fusion['final_offset'] = sample_offsets
        sample_seconds = time.perf_counter() - sample_start_time

        record = {
            'sample_index': idx_value,
            'history_times': hist_times,
            'future_times': fut_times,
            'stage1_json': stage1_json,
            'stage1_parse_ok': stage1_parse_ok,
            'stage1_cache_hit': stage1_result['cache_hit'],
            'stage2_nomemory_cache_hit': stage2_cache_hit,
            'stage1_response': stage1_response,
            'stage2_nomemory_json': no_memory_json,
            'stage2_nomemory_response': stage2_response,
            'stage2_memory_json': memory_json,
            'stage2_memory_response': memory_response,
            'stage2_memory_cache_hit': memory_cache_hit,
            'fusion': fusion,
            'no_memory_context_offset': no_memory_offsets,
            'context_offset': sample_offsets,
            'timing_seconds': {
                'stage1': stage1_seconds,
                'stage2_nomemory': stage2_seconds,
                'stage2_memory': memory_seconds,
                'sample_total': sample_seconds,
            },
        }
        if memory_retrieval is not None:
            record['memory_retrieval'] = self._compact_memory_retrieval(memory_retrieval)
        if bool(int(_config_or_default(self.configs, 'context_save_prompts', 'save_prompts'))):
            record['stage1_prompt'] = stage1_prompt
            record['stage2_nomemory_prompt'] = stage2_prompt
            record['stage2_memory_prompt'] = memory_prompt
        return row_idx, sample_offsets, record

    @staticmethod
    def _fuse_offsets(
            no_memory_offsets: np.ndarray,
            memory_offsets: np.ndarray,
            memory_retrieval: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        analog_center = memory_retrieval.get('query', {}).get('analog_offset_center')
        if analog_center is None:
            return no_memory_offsets, {
                'used_memory': False,
                'method': 'context_only',
                'reason': 'missing_analog_offset_center',
                'context_only_offset': no_memory_offsets,
                'final_offset': no_memory_offsets,
            }
        analog_center = np.asarray(analog_center, dtype=float)
        if analog_center.shape != no_memory_offsets.shape:
            return no_memory_offsets, {
                'used_memory': False,
                'method': 'context_only',
                'reason': 'analog_offset_shape_mismatch',
                'context_only_offset': no_memory_offsets,
                'final_offset': no_memory_offsets,
            }
        if not (np.all(np.isfinite(no_memory_offsets))
                and np.all(np.isfinite(memory_offsets))
                and np.all(np.isfinite(analog_center))):
            return no_memory_offsets, {
                'used_memory': False,
                'method': 'context_only',
                'reason': 'non_finite_memory_candidate',
                'context_only_offset': no_memory_offsets,
                'final_offset': no_memory_offsets,
            }

        stabilized = np.median(
            np.stack([no_memory_offsets, memory_offsets, analog_center]),
            axis=0,
        )
        return stabilized, {
            'used_memory': True,
            'method': 'rowwise_median',
            'context_only_offset': no_memory_offsets,
            'memory_conditioned_offset': memory_offsets,
            'numerical_analog_offset': analog_center,
            'final_offset': stabilized,
        }

    @staticmethod
    def _compact_memory_retrieval(retrieval: Dict[str, Any]) -> Dict[str, Any]:
        examples = []
        for example in retrieval.get('examples', []):
            entry = example.get('entry', {})
            examples.append({
                'memory_index': example.get('memory_index'),
                'score': example.get('score'),
                'text_score': example.get('text_score'),
                'history_shape_score': example.get('history_shape_score'),
                'implied_current_offset_mean': example.get('implied_current_offset_mean'),
                'sample_index': entry.get('sample_index'),
                'source_split': entry.get('source_split'),
                'observed_true_future': np.asarray(entry.get('true_future', []), dtype=float),
            })
        return {'query': retrieval.get('query', {}), 'examples': examples}

    def _parse_offsets(
            self,
            data: Dict[str, Any],
            future_times: Sequence[str],
            variables: Sequence[str],
    ) -> np.ndarray:
        offsets = np.zeros((len(future_times), len(variables)), dtype=float)
        time_to_idx = {str(time_value): idx for idx, time_value in enumerate(future_times)}
        values_obj = data.get('context_offset_values')
        if values_obj is None:
            values_obj = data.get('final_context_offset_values', {})
        if isinstance(values_obj, dict):
            offsets_by_variable = values_obj.get('offsets_by_variable')
            if isinstance(offsets_by_variable, dict):
                for var_idx, variable in enumerate(variables):
                    series = offsets_by_variable.get(variable)
                    if series is None:
                        series = offsets_by_variable.get(str(variable))
                    if series is None:
                        series = offsets_by_variable.get(f'{variable}_offset')
                    if not isinstance(series, list):
                        continue
                    if not series:
                        continue
                    padded_series = list(series[:len(future_times)])
                    if len(padded_series) < len(future_times):
                        padded_series.extend([padded_series[-1]] * (len(future_times) - len(padded_series)))
                    for time_idx, value in enumerate(padded_series):
                        offsets[time_idx, var_idx] = _safe_float(value)
                return offsets

            columns = values_obj.get('columns') or values_obj.get('variables')
            values = values_obj.get('values', [])
            if isinstance(columns, list) and isinstance(values, list):
                column_to_idx = {str(column): idx for idx, column in enumerate(columns)}
                parsed_matrix = False
                for time_idx, row in enumerate(values[:len(future_times)]):
                    if not isinstance(row, (list, tuple)):
                        parsed_matrix = False
                        break
                    parsed_matrix = True
                    for var_idx, variable in enumerate(variables):
                        col_idx = column_to_idx.get(str(variable))
                        if col_idx is None or col_idx >= len(row):
                            continue
                        offsets[time_idx, var_idx] = _safe_float(row[col_idx])
                if parsed_matrix:
                    return offsets

            if len(variables) == 1 and isinstance(values, list) and all(
                    not isinstance(row, dict) for row in values):
                if not values:
                    return offsets
                padded_values = list(values[:len(future_times)])
                if len(padded_values) < len(future_times):
                    padded_values.extend([padded_values[-1]] * (len(future_times) - len(padded_values)))
                for time_idx, value in enumerate(padded_values):
                    offsets[time_idx, 0] = _safe_float(value)
                return offsets
        elif isinstance(values_obj, list):
            values = values_obj
        else:
            values = data.get('values', [])

        if not isinstance(values, list):
            return offsets

        fallback_idx = 0
        for row in values:
            if not isinstance(row, dict):
                continue
            time_value = str(row.get('forecast_time', ''))
            time_idx = time_to_idx.get(time_value)
            if time_idx is None:
                if fallback_idx >= len(future_times):
                    continue
                time_idx = fallback_idx
            fallback_idx += 1

            row_offsets = row.get('offsets', row)
            if not isinstance(row_offsets, dict):
                continue
            for var_idx, variable in enumerate(variables):
                offsets[time_idx, var_idx] = _safe_float(
                    row_offsets.get(variable, row_offsets.get(f'{variable}_offset', 0.0))
                )
        return offsets

    def _clip_offsets(self, offsets: np.ndarray, history_values: np.ndarray, variables: Sequence[str]) -> np.ndarray:
        if self.clip_factor <= 0:
            return offsets
        scale_df = self.prompt_builder.scale_hint_df(history_values, variables)
        limits = scale_df['recent_max_abs_change'].to_numpy(dtype=float) * self.clip_factor
        limits = np.maximum(limits, 1e-8)
        return np.clip(offsets, -limits.reshape(1, -1), limits.reshape(1, -1))


class Model(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.device = device
        self.configs = configs
        self.model_dict = {
            'TimesNet': 'models.TimesNet',
            'Autoformer': 'models.Autoformer',
            'Transformer': 'models.Transformer',
            'Nonstationary_Transformer': 'models.Nonstationary_Transformer',
            'DLinear': 'models.DLinear',
            'FEDformer': 'models.FEDformer',
            'Informer': 'models.Informer',
            'LightTS': 'models.LightTS',
            'Reformer': 'models.Reformer',
            'ETSformer': 'models.ETSformer',
            'PatchTST': 'models.PatchTST',
            'Pyraformer': 'models.Pyraformer',
            'MICN': 'models.MICN',
            'Crossformer': 'models.Crossformer',
            'FiLM': 'models.FiLM',
            'iTransformer': 'models.iTransformer',
            'Koopa': 'models.Koopa',
            'TiDE': 'models.TiDE',
            'FreTS': 'models.FreTS',
            'TimeMixer': 'models.TimeMixer',
            'TSMixer': 'models.TSMixer',
            'SegRNN': 'models.SegRNN',
        }
        self.base_model_name = getattr(configs, 'context_base_model', CONTEXT_OFFSET_SETTINGS['base_model'])
        if self.base_model_name in {'AFTER', 'ContextOffset'}:
            raise ValueError('context_base_model must be a numerical forecasting model.')
        self.base_model = self._build_model().to(self.device)
        self._reasoner = None

    def _build_model(self):
        if self.base_model_name not in self.model_dict:
            raise ValueError(f'Unknown base model: {self.base_model_name}')
        module = importlib.import_module(self.model_dict[self.base_model_name])
        return module.Model(self.configs).float()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)

    def context_memory_path(self) -> str:
        return context_memory_path(self.configs)

    def build_context_memory(
            self,
            history_values: np.ndarray,
            history_times: Sequence[Sequence[str]],
            context_texts: Sequence[Sequence[str]],
            future_times: Sequence[Sequence[str]],
            true_future: np.ndarray,
            variables: Sequence[str],
            sample_indices: Sequence[Any],
            source_splits: Sequence[str],
    ) -> Dict[str, Any]:
        path = self.context_memory_path()
        rebuild = bool(int(_config_or_default(
            self.configs, 'context_memory_rebuild', 'memory_rebuild')))
        if os.path.isfile(path) and not rebuild:
            return {'path': path, 'reused': True, 'entries': None}

        history_values = np.asarray(history_values)
        true_future = np.asarray(true_future)
        candidate_pool = [
            idx for idx, texts in enumerate(context_texts)
            if _has_any_context_text(texts)
        ]
        if not candidate_pool:
            EventResponseMemory.save(path, [])
            self._reasoner = None
            return {
                'path': path,
                'reused': False,
                'entries': 0,
                'stage1_candidates': 0,
                'stage1_cache_hits': 0,
            }

        reasoner = ContextOffsetReasoner(self.configs, load_memory=False)
        stage1_results = reasoner.run_stage1_batch(
            history_values=history_values[candidate_pool],
            history_times=[history_times[idx] for idx in candidate_pool],
            context_texts=[context_texts[idx] for idx in candidate_pool],
            future_times=[future_times[idx] for idx in candidate_pool],
            variables=variables,
        )
        selected_positions = select_context_state_candidates(stage1_results)

        entries = []
        for selected_pos in selected_positions:
            source_pos = candidate_pool[selected_pos]
            stage1_result = stage1_results[selected_pos]
            entries.append({
                'sample_index': _json_default(sample_indices[source_pos]),
                'source_split': str(source_splits[source_pos]),
                'history_values': np.asarray(history_values[source_pos], dtype=float).tolist(),
                'true_future': np.asarray(true_future[source_pos], dtype=float).tolist(),
                'history_times': list(history_times[source_pos]),
                'future_times': list(future_times[source_pos]),
                'stage1_json': stage1_result['stage1_json'],
                'stage1_cache_hit': bool(stage1_result['cache_hit']),
                'experiment': {
                    'dataset': str(getattr(self.configs, 'data', '')),
                    'backbone': self.base_model_name,
                    'seed': int(getattr(self.configs, 'seed', 0)),
                    'seq_len': int(getattr(self.configs, 'seq_len', 0)),
                    'label_len': int(getattr(self.configs, 'label_len', 0)),
                    'pred_len': int(getattr(self.configs, 'pred_len', 0)),
                },
            })
        EventResponseMemory.save(path, entries)
        self._reasoner = None
        return {
            'path': path,
            'reused': False,
            'entries': len(entries),
            'stage1_candidates': len(stage1_results),
            'transferable_context_candidates': len(selected_positions),
            'stage1_cache_hits': sum(int(entry['stage1_cache_hit']) for entry in entries),
        }

    def generate_context_adjusted_forecast(
            self,
            history_values: np.ndarray,
            history_times: Sequence[Sequence[str]],
            context_texts: Sequence[Sequence[str]],
            future_times: Sequence[Sequence[str]],
            base_forecast: np.ndarray,
            variables: Sequence[str],
            sample_indices: Optional[Sequence[Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        # 延迟加载
        if self._reasoner is None:
            self._reasoner = ContextOffsetReasoner(self.configs)
        return self._reasoner.run_batch(
            history_values=history_values,
            history_times=history_times,
            context_texts=context_texts,
            future_times=future_times,
            base_forecast=base_forecast,
            variables=variables,
            sample_indices=sample_indices,
        )

    def save_context_artifacts(
            self,
            output_dir: str,
            records: Sequence[Dict[str, Any]],
            inputs: np.ndarray,
            base_preds: np.ndarray,
            offsets: np.ndarray,
            preds: np.ndarray,
            trues: np.ndarray,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'context_records.jsonl'), 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + '\n')

        np.save(os.path.join(output_dir, 'input.npy'), inputs)
        np.save(os.path.join(output_dir, 'base_pred.npy'), base_preds)
        np.save(os.path.join(output_dir, 'offset.npy'), offsets)
        np.save(os.path.join(output_dir, 'pred.npy'), preds)
        np.save(os.path.join(output_dir, 'true.npy'), trues)
