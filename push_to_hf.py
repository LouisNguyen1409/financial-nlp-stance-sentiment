"""
Push all trained models to HuggingFace Hub.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, RESULTS_DIR

from huggingface_hub import HfApi, upload_folder, upload_file

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_USERNAME = "Louisnguyen"

HF_FORMAT_MODELS = [
    ("finbert_stance", "finbert-financial-stance"),
    ("finbert_sentiment", "finbert-financial-sentiment"),
    ("bert_llrd_stance", "bert-llrd-financial-stance"),
    ("bert_llrd_sentiment", "bert-llrd-financial-sentiment"),
]

MULTITASK_MODEL = ("multitask_finbert", "multitask-finbert-financial")


def push_hf_format_model(local_name, repo_name):
    local_path = os.path.join(MODELS_DIR, local_name)
    repo_id = f"{HF_USERNAME}/{repo_name}"

    if not os.path.exists(local_path):
        print(f"  [SKIP] {local_path} not found")
        return False

    print(f"\n  Pushing {local_name} -> {repo_id}")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False, token=HF_TOKEN)

    upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        token=HF_TOKEN,
        commit_message=f"Upload {local_name} trained model",
    )

    results_files = {
        "finbert_stance": "finetune_finbert_stance.json",
        "finbert_sentiment": "finetune_finbert_sentiment.json",
        "bert_llrd_stance": "finetune_bert_llrd_stance.json",
        "bert_llrd_sentiment": "finetune_bert_llrd_sentiment.json",
    }
    if local_name in results_files:
        result_path = os.path.join(RESULTS_DIR, results_files[local_name])
        if os.path.exists(result_path):
            upload_file(
                path_or_fileobj=result_path,
                path_in_repo="results.json",
                repo_id=repo_id,
                token=HF_TOKEN,
                commit_message="Upload evaluation results",
            )

    print(f"  [OK] {repo_id} pushed successfully")
    return True


def push_multitask_model(local_name, repo_name):
    local_path = os.path.join(MODELS_DIR, local_name)
    repo_id = f"{HF_USERNAME}/{repo_name}"

    if not os.path.exists(local_path):
        print(f"  [SKIP] {local_path} not found")
        return False

    print(f"\n  Pushing {local_name} -> {repo_id}")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False, token=HF_TOKEN)

    upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        token=HF_TOKEN,
        commit_message=f"Upload {local_name} multi-task model",
    )

    for task in ["stance", "sentiment"]:
        result_path = os.path.join(RESULTS_DIR, f"multitask_{task}.json")
        if os.path.exists(result_path):
            upload_file(
                path_or_fileobj=result_path,
                path_in_repo=f"results_{task}.json",
                repo_id=repo_id,
                token=HF_TOKEN,
                commit_message=f"Upload {task} evaluation results",
            )

    print(f"  [OK] {repo_id} pushed successfully")
    return True


def main():
    print("=" * 60)
    print("  PUSHING MODELS TO HUGGINGFACE HUB")
    print("=" * 60)

    success_count = 0
    for local_name, repo_name in HF_FORMAT_MODELS:
        if push_hf_format_model(local_name, repo_name):
            success_count += 1

    local_name, repo_name = MULTITASK_MODEL
    if push_multitask_model(local_name, repo_name):
        success_count += 1

    total = len(HF_FORMAT_MODELS) + 1
    print(f"\n{'=' * 60}")
    print(f"  DONE: {success_count}/{total} models pushed to HuggingFace Hub")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
