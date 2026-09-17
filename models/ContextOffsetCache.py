import hashlib
import json
import os
import threading
from typing import Any, Callable, Dict, Tuple


class PromptResponseCache:
    def __init__(self, cache_dir: str, namespace: Dict[str, Any], enabled: bool = True):
        self.cache_dir = str(cache_dir)
        self.namespace = dict(namespace)
        self.enabled = bool(enabled)
        self._lock = threading.Lock()

    def get_or_generate(
            self,
            stage: str,
            prompt: str,
            max_new_tokens: int,
            generate: Callable[[], str],
    ) -> Tuple[str, bool]:
        cached, cache_path = self.lookup(
            stage=stage,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        if cached is not None:
            return cached, True

        response = generate()
        if cache_path is None:
            return str(response), False

        with self._lock:
            cached = self._read(cache_path)
            if cached is not None:
                return cached, True
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            temporary = cache_path + f'.{os.getpid()}.{threading.get_ident()}.tmp'
            with open(temporary, 'w', encoding='utf-8') as handle:
                json.dump(
                    {
                        'cache_key': os.path.basename(cache_path)[:-5],
                        'stage': str(stage),
                        'max_new_tokens': int(max_new_tokens),
                        'response': str(response),
                    },
                    handle,
                    ensure_ascii=False,
                )
            os.replace(temporary, cache_path)
        return str(response), False

    def lookup(self, stage: str, prompt: str, max_new_tokens: int):
        """Return a cached response and its path without generating a response."""
        if not self.enabled:
            return None, None

        key_payload = {
            'namespace': self.namespace,
            'stage': str(stage),
            'max_new_tokens': int(max_new_tokens),
            'prompt': str(prompt),
        }
        serialized = json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
        cache_key = hashlib.sha256(serialized).hexdigest()
        cache_path = os.path.join(self.cache_dir, cache_key[:2], cache_key + '.json')
        return self._read(cache_path), cache_path

    @staticmethod
    def _read(path: str):
        if not os.path.isfile(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            response = payload.get('response')
            return str(response) if response is not None else None
        except (OSError, ValueError, TypeError):
            return None
