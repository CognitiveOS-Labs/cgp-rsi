"""Precision-preserving weight pipeline (hardware-gated stage).

Master checkpoint policy: M_t^{FP16|BF16} (.safetensors) is canonical.
LoRA adapters are merged in FP16, and ONLY the merged FP16 weights are
quantized to GGUF for deployment. LoRA MUST NOT be merged into quantized
weights across generations (it compounds quantization drift).

These functions import heavy deps lazily and raise HardwareStageUnavailable
with actionable instructions when the training host is not provisioned.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


class HardwareStageUnavailable(RuntimeError):
    pass


@dataclass
class MergeResult:
    merged_safetensors: Path
    adapter_norm: float


def _require(module: str, hint: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise HardwareStageUnavailable(
            f"optional dependency '{module}' is not installed. {hint}"
        ) from exc


def merge_lora(base_dir: str | Path, adapter_dir: str | Path, out_dir: str | Path) -> MergeResult:
    """Fuse a LoRA adapter into the FP16 master and write safetensors."""
    peft = _require("peft", "Run on the GPU training host: pip install peft cpm safetensors")
    st = _require("safetensors", "Run on the GPU training host: pip install safetensors")
    torch = _require("torch", "Run on the GPU training host: pip install torch")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    adapter_norm = float(
        torch.sqrt(
            sum((p.detach().float() ** 2).sum() for p in model.parameters() if p.requires_grad)
        ).item()
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)
    # Explicit FP16 check: refuse to write a non-FP16 master.
    for f in out.glob("*.safetensors"):
        from safetensors import safe_open

        with safe_open(str(f), framework="pt") as h:
            dtypes = {h.get_slice(k).get_dtype() for k in h.keys()}
        if dtypes - {"F16", "BF16"}:
            raise HardwareStageUnavailable(
                f"merged master is not FP16/BF16 (found {sorted(dtypes)}); refusing to proceed"
            )
    return MergeResult(merged_safetensors=out, adapter_norm=adapter_norm)


def export_gguf(merged_dir: str | Path, out_file: str | Path, quant: str = "Q8_0") -> Path:
    """Quantize the merged FP16 master to GGUF via llama.cpp."""
    script = os.environ.get("LLAMA_CPP_EXPORT_SCRIPT")
    if not script:
        raise HardwareStageUnavailable(
            "env LLAMA_CPP_EXPORT_SCRIPT not set. Point it at llama.cpp/convert_hf_to_gguf.py "
            "on the GPU host (see docs/REPRODUCIBILITY.md)."
        )
    out = Path(out_file)
    cmd = ["python3", script, str(merged_dir), "--outfile", str(out), "--outtype", quant.lower()]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise HardwareStageUnavailable(f"GGUF export failed:\n{proc.stderr[-2000:]}")
    return out


def quantization_drift(fp16_weight_path: str | Path, gguf_sha_note: str = "") -> dict:
    """Record quantization-drift evidence for the report.

    Full parity probing (same prompts, logit comparison FP16 vs GGUF) is a
    GPU-host step; here we emit the artifact manifest the analysis consumes.
    """
    p = Path(fp16_weight_path)
    return {
        "fp16_bytes": p.stat().st_size if p.exists() else None,
        "note": "Logit-parity delta (KL / top-1 agreement) MUST be measured on the GPU host; "
                "store results under runs/quantization-drift.json." + gguf_sha_note,
    }
