import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import engine as E
from tokenizer import Qwen2Tokenizer

_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_model_dir() -> str:
    env = os.environ.get("QWEN_MODEL_DIR")
    if env:
        return env

    for name in ("Qwen2.5-0.5B", "qwen2-model"):
        for d in (os.path.join(_DIR, name), os.path.join(os.path.dirname(_DIR), name)):
            if os.path.exists(os.path.join(d, "model.safetensors")):
                return d

    repo_id = os.environ.get("QWEN_HF_REPO", "Qwen/Qwen2.5-0.5B")
    token = os.environ.get("QWEN_HF_TOKEN") or os.environ.get("HF_TOKEN")

    from huggingface_hub import snapshot_download
    print(f"[run.py] مفيش موديل لوكال، جاري التحميل من HF: {repo_id}")
    return snapshot_download(
        repo_id=repo_id,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json"],
        token=token,
    )


MODEL_DIR = _find_model_dir()


def main() -> None:
    tok = Qwen2Tokenizer.from_pretrained(MODEL_DIR)
    eng = E.Engine.load(MODEL_DIR)

    args = sys.argv[1:]
    prompt = " ".join(args[:-1]) or "The capital of France is"
    max_new = int(args[-1]) if args and args[-1].isdigit() else 30

    ids = torch.tensor([tok.encode(prompt)], device="cuda")
    out = eng.generate(ids, max_new_tokens=max_new, eos_token_id=tok.eos_token_id)
    print(f"[prompt] {prompt}")
    print(tok.decode(out, skip_special_tokens=True))


if __name__ == "__main__":
    main()
