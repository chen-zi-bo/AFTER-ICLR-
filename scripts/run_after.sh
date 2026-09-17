#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

gpu_id="${1:-0}"
dataset_spec="${2:-all}"
backbone="${BACKBONE:-iTransformer}"
seed="${SEED:-1}"
memory_name="${AFTER_MEMORY_NAME:-event_response_memory_v1}"
output_root="${AFTER_OUTPUT_ROOT:-./results/after_deepseek}"
log_dir="${LOG_DIR:-./logs/after_deepseek}"

all_datasets=(agriculture climate economy energy environment health security socialgood traffic)
if [[ "${dataset_spec}" == "all" ]]; then
  datasets=("${all_datasets[@]}")
else
  IFS=',' read -r -a datasets <<< "${dataset_spec}"
fi

pred_lengths() {
  case "$1" in
    energy|health) echo "4 8 12 24" ;;
    environment) echo "7 14 30 48" ;;
    *) echo "6 8 10 12" ;;
  esac
}

window_lengths() {
  case "$1" in
    environment) echo "48 24" ;;
    *) echo "24 12" ;;
  esac
}

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY or copy .env.example to .env}"
api_model="${DEEPSEEK_MODEL:-deepseek-v4-flash}"

mkdir -p "${output_root}" "${log_dir}"
export TOKENIZERS_PARALLELISM=false

for dataset in "${datasets[@]}"; do
  read -r seq_len label_len <<< "$(window_lengths "${dataset}")"
  for pred_len in $(pred_lengths "${dataset}"); do
    job_name="AFTER_${backbone}_${dataset}_s${seed}_sl${seq_len}_ll${label_len}_pl${pred_len}"
    summary_path="${log_dir}/${job_name}.txt"
    checkpoint="./checkpoints/AFTER_${backbone}_${dataset}_${seed}_${seq_len}_${label_len}_${pred_len}_1/checkpoint.pth"
    is_training=0
    [[ -s "${checkpoint}" ]] || is_training=1

    echo "[$(date '+%F %T')] dataset=${dataset} pred_len=${pred_len} backend=deepseek"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python -u run.py \
      --task_name long_term_forecast \
      --is_training "${is_training}" \
      --model_id "${job_name}" \
      --model AFTER \
      --context_base_model "${backbone}" \
      --data "${dataset}" \
      --seq_len "${seq_len}" \
      --label_len "${label_len}" \
      --pred_len "${pred_len}" \
      --seed "${seed}" \
      --gpu 0 \
      --resume 1 \
      --train_epochs "${TRAIN_EPOCHS:-10}" \
      --batch_size "${BATCH_SIZE:-32}" \
      --learning_rate "${LEARNING_RATE:-0.0001}" \
      --num_workers "${NUM_WORKERS:-4}" \
      --checkpoints ./checkpoints \
      --save_name "${summary_path}" \
      --des after_deepseek \
      --context_llm_backend deepseek \
      --context_api_model "${api_model}" \
      --context_api_key_env DEEPSEEK_API_KEY \
      --context_llm_temperature 0 \
      --context_llm_top_p 1 \
      --context_api_concurrency "${API_CONCURRENCY:-1}" \
      --context_api_response_format json_object \
      --context_api_timeout "${API_TIMEOUT:-600}" \
      --context_api_max_retries "${API_MAX_RETRIES:-4}" \
      --context_output_dir "${output_root}" \
      --context_memory_enabled 1 \
      --context_memory_name "${memory_name}" \
      --context_memory_rebuild "${MEMORY_REBUILD:-0}"
  done
done

echo "AFTER experiments finished. Results: ${output_root}"
