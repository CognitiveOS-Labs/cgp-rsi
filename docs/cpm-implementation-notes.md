# cpm implementation notes (v1.0.9-alpha)

Findings from building this repo against the current `cpm` implementation and
the draft specs, mirroring `face-recognition-tune`'s experience. **The shipping
implementation wins over the specs for what `cpm pack` accepts.**

1. **`brain.training`** — the field `cpm tune` reads (per the `cpm-tune.md`
   tutorial). It is **not present** in either schema (the local mirror
   `schemas/cognitive.schema.json` has top-level `training` instead; cpm's
   bundled schema seemingly has neither). It passes only because unknown
   properties are allowed. It is accepted but **not schema-validated**. This
   repo sets it in **both** packages (same as `face-recognition-tune`).
2. **Top-level `training`** exists in the canonical schema mirror
   (`schemas/cognitive.schema.json`) but is **not** duplicated in these
   manifests — the implementation reads `brain.training`, so we follow the
   implementation and document the mirror divergence here.
3. **`dependencies`** — cpm's schema validator rejects the field for any value
   (`ERROR:V006 ... invalid jsonType map[string]string`). The required
   `rsi-controller` ↔ `base-model-v0` pairing is documented in READMEs and
   descriptions instead of declared in the manifest. Re-add
   `"dependencies": {"rsi-controller": "^0.1.0"}` once cpm accepts the field.
4. **`brain.wide_model.routing`** — the spec defines `{model_id, tags}`; the
   cpm struct defines an array of `{capability, priority}`. Divergent
   semantics → `routing` is omitted.
5. **Tools exec bit** — plain `cpm pack` stores `tools/` at `0644`. Use
   **`cpm pack --bin tools`** so `tools/rsi-controller` (a `#!` python
   script) stays `0755`. Verified: `cpm pack --bin tools` + `cpm verify` ✓.
6. **`cpm pack` for manifest-only models** — `cpm pack --manifest
   cognitive.json` packages just the manifest (remote weights are not local
   files); `cpm verify` passes on the archive.
7. **`cpm tune <name>`** operates on an *installed* patch and reads its
   `brain.training` block; it invokes the designated MCP tool through the
   system daemon. Our MCP server exposes `cognitiveos.rsi.tune` and refuses
   with `E_HW_UNAVAILABLE` on hosts without the GPU stack — fail-closed, never
   a fake training run.
8. **Hot-swap endpoint** (`/api/adapter`) — assumed by `cpm tune` docs; not
   yet documented in `inference-api.md`. Not relied on for correctness here:
   promotion is defined at the `.cgp` artifact level (install/uninstall), not
   at the inference-engine adapter-swap level.

Manifest verification log (2026-11-17, this repo):

```text
packages/base-model-v0    cpm pack --manifest cognitive.json  → ✓ valid
packages/rsi-controller   cpm pack --bin tools                → ✓ valid (0755 tool)
both                      cpm verify                          → ✓ OK
```
