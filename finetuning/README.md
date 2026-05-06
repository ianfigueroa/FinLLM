# Fine-Tuning Experiments

This project generates RAFT-style data, but it does not train a real model by default.

The local app can export LoRA-style JSONL records from indexed filings. Each JSONL line is
an instruction-tuning record with:

- a system instruction that requires evidence-grounded financial answers;
- a user message containing relevant chunks plus distractor chunks;
- an assistant message containing the cited answer.

## Export

In the UI, index documents and click `Export JSONL`. The browser downloads:

```text
finllm-raft-lora.jsonl
```

The API route is:

```text
POST /api/v1/finetuning/raft/export
```

The export is intentionally separate from training. That keeps the local app deterministic,
cheap to run, and easy to evaluate.

## Why RAFT

RAFT-style data teaches a model to answer from supplied documents while ignoring distractor
chunks. For financial research, the target behavior is:

- use the filing evidence;
- cite the evidence;
- ignore irrelevant context;
- say when evidence is insufficient.

## Real Training Paths

OpenAI fine-tuning is the simplest managed path, but it costs money and requires converting
the JSONL to the exact model-specific fine-tuning format.

Local LoRA training usually requires Linux or WSL with CUDA, plus a GPU with enough VRAM for
the selected base model. A practical stack would use:

- `transformers`
- `datasets`
- `peft`
- `trl`
- `accelerate`
- optionally `bitsandbytes`

Example shape:

```powershell
py -3 finetuning/train_lora.py --dataset data/finllm-raft-lora.jsonl --base-model Qwen/Qwen2.5-7B-Instruct
py -3 finetuning/evaluate_finetuned.py --adapter outputs/finllm-lora
```

The current `train_lora.py` remains simulated until real GPU dependencies and a model license
are added.

## Evaluation Before Training

Do not assume fine-tuning improves the system. Compare:

- baseline RAG;
- RAG plus reranking;
- RAG plus self-verification;
- fine-tuned generation, if trained.

Use citation correctness, retrieval recall, hallucination proxy, latency, and cost. A trained
model is only better if it improves held-out quality without weakening citation faithfulness.
