# RevUtil Gradio Demo

This folder hosts a lightweight, locally hosted Gradio experience for inspecting RevUtil review analyses. The UI takes stylistic inspiration from the [Gemini board](https://gemini.google.com/share/3cc5d64d3705) while surfacing the model, paper, and datasets that power RevUtil.

## Features

- Single-textbox workflow: paste a reviewer comment and receive aspect-level scores plus rationales.
- Configurable backend: choose any LoRA adapter/base model combo compatible with `vLLM`.
- Mock mode: preview the UX without loading GPU-heavy weights.
- Prompt transparency: inspect the exact instruction prompt used for every request.

## Prerequisites

- Python environment with the root `requirements.txt` installed (`gradio` is already listed).
- GPU + CUDA for real inference, or set `--mock` / `REVUTIL_MOCK=1` for CPU-only previews.
- Access to the RevUtil weights on Hugging Face (default: `boda/RevUtil_Llama-3.1-8B-Instruct_score_rationale`).

## Quickstart

```bash
cd demo
./run_demo.sh  # launches on http://localhost:7860
```

Pass flags to override defaults:

```bash
./run_demo.sh --adapter boda/CustomAdapter --base-model meta-llama/Llama-3.1-8B-Instruct --mock
```

Behind the scenes this wraps `python app.py --host 0.0.0.0 --port 7860 ...`.

## Configuration knobs

| Flag | Description | Default |
| --- | --- | --- |
| `--adapter` | LoRA adapter model ID | `boda/RevUtil_Llama-3.1-8B-Instruct_score_rationale` |
| `--base-model` | Base instruction-tuned model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `--mock` | Skip model loading, return deterministic demo output | `false` |
| `--share` | Enable Gradio sharing tunnel | `false` |

Environment toggles:

- `REVUTIL_MOCK=1` — forces mock mode regardless of CLI flag.
- `REVUTIL_TENSOR_PARALLEL` / `REVUTIL_GPU_UTIL` — fine-tune vLLM resource usage.

## Reference material

- Paper: [https://arxiv.org/html/2509.04484v3](https://arxiv.org/html/2509.04484v3)
- Human data: [https://huggingface.co/datasets/boda/RevUtil_human](https://huggingface.co/datasets/boda/RevUtil_human)
- Synthetic data: [https://huggingface.co/datasets/boda/RevUtil_synthetic](https://huggingface.co/datasets/boda/RevUtil_synthetic)

## Troubleshooting

- **ImportError: gradio** – reinstall root dependencies (`pip install -r requirements.txt`).
- **vLLM GPU errors** – try reducing `REVUTIL_GPU_UTIL`, or run with `--mock` to validate the UI.
- **Adapter auth issues** – ensure `huggingface-cli login` has access to the configured model IDs.

Feel free to tailor the Blocks layout in `app.py` if you’d like to embed more datasets, charts, or canned examples.
