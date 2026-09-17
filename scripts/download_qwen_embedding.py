from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = 'Qwen/Qwen3-Embedding-0.6B'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = PROJECT_ROOT / 'llm' / 'Qwen3-Embedding-0.6B'


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
    )
    print(f'Qwen embedding model is ready at: {LOCAL_DIR}')


if __name__ == '__main__':
    main()
