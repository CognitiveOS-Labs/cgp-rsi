# Reproducing the RSI experiment

This runbook covers the full multi-generation experiment on a GPU training
host. Everything except Sections 3–6 (neural stages) can also run on a laptop;
the CI mock run (`scripts/mock_e2e.py`) exercises it without a GPU.

## 0. Hardware (minimum practical)

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 1 × 24 GB (LoRA on 1–4B model, FP16 master) | 1 × 48 GB+ or 2 × 24 GB |
| RAM | 64 GB | 128 GB |
| Disk | 200 GB (base + 6 merged masters + GGUF exports) | NVMe |
| Toolchain | `cpm >= v1.0.9-alpha`, Python ≥ 3.10, `torch`, `peft`, `safetensors`, `transformers`, llama.cpp (`convert_hf_to_gguf.py`) | same |

## 1. Toolchain setup

```sh
go install github.com/CognitiveOS-Project/cpm/cmd/cpm@v1.0.9-alpha
cpm --help
pip install -e '.[dev]'          # this repo: verifier, gate, controller, mock e2e
pip install torch peft safetensors transformers datasets
```

## 2. Install the base model patch

```sh
cd packages/base-model-v0 && cpm pack --manifest cognitive.json && cpm verify base-model-v0-*.cgp
cpm install base-model-v0-0.1.0-linux-amd64.cgp          # downloads remote safetensors at install
cd ../rsi-controller && cpm pack --bin tools && cpm verify rsi-controller-*.cgp
cpm install rsi-controller-0.1.0-linux-amd64.cgp
```

## 3. Serve the current model (OpenAI-compatible)

Point an OpenAI-compatible server at the current master weights, e.g.
CognitiveOS `code-inference` (`start.sh`, `http://api:8000/v1`) or vLLM with
`--dtype float16 --max-model-len 16384`. Set:

```sh
export RSI_BASE_URL=http://api:8000/v1
export RSI_MODEL=model.gguf            # or the served model id
export RSI_API_KEY=                    # if the endpoint requires one
```

## 4. Baselines and generation loop

```sh
python3 scripts/run_generation.py --generations 6 --samples-per-gen 64
```

`run_generation.py` first measures `B_target`/`B_general` for the *current*
endpoint (the M_0 baseline), then iterates
`Generate → Verify → Mix → cpm tune → Merge → Export → Evaluate → Gate`.

**Promote step (per generation, after a pass):** fuse the adapter into the
master (`rsi/merge.py`), quantize to GGUF, publish/verify
`base-model-v{t+1}.cgp` (CI `.github/workflows/release.yml` attaches it to a
GitHub Release), then re-point the inference endpoint at the new artifact and
continue to generation t+1. **On gate failure:** the controller calls
`cpm install <previous .cgp>`; the next attempt uses the adjusted
hyperparameters (`gate.suggest_hyperparameter_adjustment`).

Per-generation metrics land in `runs/metrics.jsonl`:

```json
{"generation": 0, "synthetic_pass_rate": ..., "train_loss": ..., "lora_weight_norm": ...,
 "b_target_new": ..., "b_target_base": ..., "b_general_new": ..., "b_general_base": ...,
 "target_margin": ..., "general_delta": ..., "promoted": true, "failure_mode": null,
 "inference_latency_ms": ..., "rollback": false}
```

## 5. Ablation study

```sh
python3 scripts/run_ablation.py A   # full recursive loop (t=0..5)
python3 scripts/run_ablation.py B   # single LoRA over M_0, equivalent total compute/rank,
                                    # trained only on M_0-generated verified data
python3 scripts/run_ablation.py C   # full loop WITHOUT the verifier (verifier verdicts ignored)
```

All arms MUST use the same benchmarks, seeds (`RSI_SEED`), anchor buffer and
compute budget so the comparison in `RSI_CGP_PROOF.md` is valid.

## 6. Quantization drift measurement (per generation)

```sh
llama.cpp convert_hf_to_gguf.py <merged_dir> --outfile M.gguf --outtype q8_0
```

then measure top-1 agreement / KL divergence between FP16 master and GGUF on
`B_target` prompts; store in `runs/quantization-drift.json`. Acceptance: the
promoted GGUF must score within noise of the FP16 master on `B_target`
(report actual numbers; do not assume).

## 7. Reporting

`runs/metrics.jsonl` (+ `quantization-drift.json`) are the raw inputs for
`experiments/RESULTS.md` in the research repo and for the paper tables. Copy
them there verbatim; never hand-edit.

## Determinism & seeds

`RSI_SEED` (default 1337) seeds synthesis, mixing and benchmark evaluation.
The verifier is deterministic; training itself is not bitwise deterministic on
GPU (non-deterministic kernels) — report variance across ≥ 2 seeds if you
need error bars, and keep the seed list in `experiments/PROTOCOL.md`.
