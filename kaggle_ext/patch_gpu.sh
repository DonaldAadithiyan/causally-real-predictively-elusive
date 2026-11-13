#!/usr/bin/env bash
# Repoint the mechanistic-conformal extension scripts at the GPU.
#
# Run ONCE from the repo root on Kaggle, before launching any experiment:
#     bash patch_gpu.sh
#
# It rewrites the handful of hard-coded "cpu" device strings to read
# ${MC_DEVICE:-cuda}. Purely a device change — no scientific-path edits.
# Idempotent: re-running is a no-op (the sentinels are already in place).
#
# To undo: `git checkout experiments/` (or restore from the pristine copy the
# notebook keeps).
set -euo pipefail

say() { printf '[patch_gpu] %s\n' "$*"; }

# helper: portable in-place sed
sedi() { if sed --version >/dev/null 2>&1; then sed -i "$@"; else sed -i '' "$@"; fi; }

DEV_EXPR='os.environ.get("MC_DEVICE", "cuda")'

# --- 1. the three scripts with `    dev = "cpu"` in run_one()/main() -------------
for f in experiments/phase3_real_predictability.py \
         experiments/phase4_oracle_ceiling.py \
         experiments/arch_gen_timemixer.py; do
    if grep -q 'dev = "cpu"' "$f"; then
        sedi "s|    dev = \"cpu\"|    import os as _os; dev = _os.environ.get(\"MC_DEVICE\", \"cuda\")  # PATCHED|" "$f"
        say "patched dev= in $f"
    else
        say "already patched: $f"
    fi
done

# --- 2. phase5_sae_reproducibility.py: several device="cpu" literals -----------
f=experiments/phase5_sae_reproducibility.py
if grep -q 'device="cpu"' "$f"; then
    # add a module-level _DEV once, right after the last import line
    if ! grep -q '_DEV = ' "$f"; then
        sedi "s|^from validation.dose_response import sweep_pool|from validation.dose_response import sweep_pool\nimport os as _os_mc\n_DEV = _os_mc.environ.get(\"MC_DEVICE\", \"cuda\")  # PATCHED|" "$f"
    fi
    sedi 's|device="cpu"|device=_DEV|g' "$f"
    say "patched device= literals in $f"
else
    say "already patched: $f"
fi

# --- 3. config-driven scripts: flip device in the YAML the notebook copies -----
if [ -f configs/v2_phase3_h6.yaml ]; then
    sedi 's|^device: cpu|device: cuda  # PATCHED|' configs/v2_phase3_h6.yaml
    say "configs/v2_phase3_h6.yaml device -> cuda"
fi

# --- 4. device-fragile inline FVU diagnostic (3 identical lines) ---------------
# `torch.from_numpy(sae.standardize(torch.from_numpy(x)).numpy())` breaks when the
# SAE buffers are on cuda/mps (mixed-device subtract, then .numpy() on a GPU tensor).
# Rewrite to a CPU-safe form: standardize on the SAE's own device, then .cpu().
OLD='xb = torch.from_numpy(sae.standardize(torch.from_numpy(tr_act.astype(np.float32))).numpy())'
NEW='xb = sae.standardize(torch.from_numpy(tr_act.astype(np.float32)).to(next(sae.parameters()).device)).detach()  # PATCHED'
for f in experiments/phase3_real_predictability.py \
         experiments/arch_gen_timemixer.py \
         experiments/phase5_sae_reproducibility.py; do
    if grep -qF "$OLD" "$f"; then
        # escape for sed
        esc_old=$(printf '%s\n' "$OLD" | sed 's/[.[\*^$/]/\\&/g')
        esc_new=$(printf '%s\n' "$NEW" | sed 's/[&/]/\\&/g')
        sedi "s/$esc_old/$esc_new/" "$f"
        say "patched inline FVU diagnostic in $f"
    fi
done

say "done. verify with:  grep -rn MC_DEVICE experiments/ ; grep '^device' configs/v2_phase3_h6.yaml"
