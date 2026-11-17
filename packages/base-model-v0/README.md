# base-model-v0 (.cgp)

Initial base model `M_0` for the Recursive Self-Improvement (RSI) pipeline
(`RSI_CGP_PROOF.md` in the parent repo / research repo).

- **Weights:** remote, pulled at install time from Hugging Face
  (`google/gemma-3-1b-it`, `safetensors`, FP16/BF16 master). They are **not**
  vendored into this archive.
- **Precision policy:** this patch is the canonical FP16/BF16 master.
  `cpm tune` produces LoRA adapters; adapters are fused into the FP16 master
  and only then quantized to GGUF for `base-model-v{t+1}.cgp`. LoRA is never
  merged into quantized weights.
- **Training tool:** `brain.training.tool = cognitiveos.rsi.tune`, provided by
  the `rsi-controller` patch. `cpm tune base-model-v0` fine-tunes a LoRA
  adapter over this base. The `dependencies` manifest field is intentionally
  not used (cpm's bundled schema validator rejects it — see
  `docs/cpm-implementation-notes.md`); the `rsi-controller` requirement is
  documented here instead.
- **Benchmark isolation:** `B_target` / `B_general` live in
  `benchmarks/` of the repo and are never shipped inside the training mix.

Build: `cpm pack . --manifest cognitive.json` from this directory (or `make build` at repo root).
Verify: `cpm verify base-model-v0-0.1.0.cgp`.
