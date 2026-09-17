import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


EMBEDDING_TASK = (
    'Given a contextual forecasting state, retrieve a past case with a similar external-event '
    'mechanism, event phase, target pressure, and scope.'
)

MIN_RETRIEVED_EXAMPLES = 2


def context_state_text(stage1_json: Dict[str, Any]) -> str:
    status = str(stage1_json.get('context_status', '')).strip().lower()
    history_summary = str(stage1_json.get('history_state_summary', '')).strip()
    context_summary = str(stage1_json.get('context_summary', '')).strip()
    parts = []
    if status:
        parts.append(f'context status: {status}')
    if history_summary:
        parts.append(f'historical state: {history_summary}')
    if context_summary:
        parts.append(f'contextual state: {context_summary}')
    return '\n'.join(parts)


def retrieval_context_text(stage1_json: Dict[str, Any]) -> str:
    return str(stage1_json.get('context_summary', '')).strip()


def has_transferable_context_state(stage1_json: Dict[str, Any]) -> bool:
    status = str(stage1_json.get('context_status', '')).strip().lower()
    return status in {'weak', 'active'} and bool(retrieval_context_text(stage1_json))


def select_context_state_candidates(
        stage1_results: Sequence[Dict[str, Any]],
) -> List[int]:
    return [
        stage1_pos
        for stage1_pos, result in enumerate(stage1_results)
        if has_transferable_context_state(result.get('stage1_json', {}))
    ]


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2:
        raise ValueError(f'Expected a 1D or 2D time series, got shape {values.shape}.')
    return values


def _history_change_shape(history: np.ndarray) -> np.ndarray:
    history = _ensure_2d(history)
    history_steps = np.diff(history, axis=0)
    if not len(history_steps):
        return np.zeros(0, dtype=float)

    step_scale = np.median(np.abs(history_steps), axis=0)
    fallback_scale = np.std(history_steps, axis=0)
    step_scale = np.where(step_scale > 1e-6, step_scale, fallback_scale)
    step_scale = np.where(step_scale > 1e-6, step_scale, 1.0)
    return np.clip(history_steps / step_scale, -5.0, 5.0).reshape(-1)


def _shape_similarity(query: np.ndarray, candidate: np.ndarray) -> float:
    if query.shape != candidate.shape or not query.size:
        return 0.0
    rmse = float(np.sqrt(np.mean((query - candidate) ** 2)))
    return float(np.exp(-rmse))


def _entry_window_times(entry: Dict[str, Any]) -> set:
    return {
        str(time_value)
        for key in ('history_times', 'future_times')
        for time_value in entry.get(key, [])
    }


class QwenTextEmbedder:
    def __init__(self, model_path: str, max_length: int = 2048, batch_size: int = 8):
        path = Path(os.path.expanduser(str(model_path)))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        self.model_path = path.resolve()
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self._lock = threading.Lock()

        weight_files = (
            list(self.model_path.glob('*.safetensors'))
            + list(self.model_path.glob('pytorch_model*.bin'))
        )
        if not self.model_path.is_dir() or not weight_files:
            raise FileNotFoundError(
                f'Qwen embedding weights are missing at {self.model_path}. '
                'Run: python scripts/download_qwen_embedding.py'
            )

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'Qwen memory retrieval requires torch and transformers>=4.51.0. '
                'Install requirements-server.txt in the inference environment.'
            ) from exc

        self.torch = torch
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                padding_side='left',
                local_files_only=True,
            )
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                torch_dtype=dtype,
                local_files_only=True,
            ).to(self.device)
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                'Failed to load Qwen3-Embedding. Ensure transformers>=4.51.0 is installed.'
            ) from exc
        self.model.eval()

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts))

    def encode_query(self, text: str) -> np.ndarray:
        instructed = f'Instruct: {EMBEDDING_TASK}\nQuery: {text}'
        return self._encode([instructed])[0]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = []
        with self._lock, self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = list(texts[start:start + self.batch_size])
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                ).to(self.device)
                outputs = self.model(**batch)
                pooled = self._last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.float().cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def _last_token_pool(self, hidden_states, attention_mask):
        if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
            return hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = self.torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, sequence_lengths]


class EventResponseMemory:
    def __init__(
            self,
            entries: Sequence[Dict[str, Any]],
            embedding_model_path: str,
            top_k: int = 3,
            text_embedder=None,
    ):
        self.entries = [dict(entry) for entry in entries]
        self.top_k = max(MIN_RETRIEVED_EXAMPLES, int(top_k))
        self._retrieval_texts = [
            retrieval_context_text(entry.get('stage1_json', {}))
            for entry in self.entries
        ]
        self._history_shapes = [
            _history_change_shape(entry['history_values'])
            for entry in self.entries
        ]
        self.embedder = text_embedder
        self._text_embeddings = np.zeros((0, 1), dtype=float)
        if self._retrieval_texts:
            if self.embedder is None:
                self.embedder = QwenTextEmbedder(embedding_model_path)
            self._text_embeddings = self.embedder.encode_documents(self._retrieval_texts)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'EventResponseMemory':
        entries = []
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    entries.append(json.loads(line))
        return cls(entries, **kwargs)

    @staticmethod
    def save(path: str, entries: Sequence[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        temporary = path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
        os.replace(temporary, path)

    def retrieve(
            self,
            history_values: np.ndarray,
            base_forecast: np.ndarray,
            stage1_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        query_text = retrieval_context_text(stage1_json)
        if not self.entries or not has_transferable_context_state(stage1_json):
            return {
                'examples': [],
                'query': {
                    'context_state_text': context_state_text(stage1_json),
                    'retrieval_context_text': query_text,
                    'reason': 'no_transferable_current_context_state',
                },
            }

        query_history_shape = _history_change_shape(history_values)
        history_scores = np.asarray([
            _shape_similarity(query_history_shape, candidate)
            for candidate in self._history_shapes
        ])
        query_embedding = self.embedder.encode_query(query_text)
        text_scores = np.clip(self._text_embeddings @ query_embedding, 0.0, 1.0)
        combined_scores = np.sqrt(text_scores * history_scores)

        selected_indices = []
        selected_windows = []
        for candidate_idx in np.argsort(combined_scores)[::-1]:
            candidate_idx = int(candidate_idx)
            candidate_window = _entry_window_times(self.entries[candidate_idx])
            if candidate_window and any(
                    candidate_window.intersection(selected_window)
                    for selected_window in selected_windows):
                continue
            selected_indices.append(candidate_idx)
            selected_windows.append(candidate_window)
            if len(selected_indices) >= self.top_k:
                break

        examples = []
        for selected_idx in selected_indices:
            entry = self.entries[selected_idx]
            memory_history = _ensure_2d(np.asarray(entry['history_values'], dtype=float))
            memory_true = _ensure_2d(np.asarray(entry['true_future'], dtype=float))
            observed_continuation = memory_true - memory_history[-1]
            analogous_current_future = (
                _ensure_2d(np.asarray(history_values, dtype=float))[-1]
                + observed_continuation
            )
            current_base = _ensure_2d(np.asarray(base_forecast, dtype=float))
            implied_current_offset = analogous_current_future - current_base
            examples.append({
                'memory_index': int(selected_idx),
                'score': float(combined_scores[selected_idx]),
                'text_score': float(text_scores[selected_idx]),
                'history_shape_score': float(history_scores[selected_idx]),
                'implied_current_offset': implied_current_offset,
                'implied_current_offset_mean': float(np.mean(implied_current_offset)),
                'entry': entry,
            })

        query = {
            'context_state_text': context_state_text(stage1_json),
            'retrieval_context_text': query_text,
            'top_score': float(np.max(combined_scores)) if combined_scores.size else 0.0,
            'ranked_candidate_count': len(self.entries),
        }
        if not examples:
            query['reason'] = 'no_non_overlapping_memory_examples'
        else:
            query['analog_offset_center'] = np.median(
                np.stack([example['implied_current_offset'] for example in examples]),
                axis=0,
            )
        return {'examples': examples, 'query': query}
