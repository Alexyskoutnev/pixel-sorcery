# Synthetic Data Generator (Real Estate Before/After)

This folder contains a small CLI tool to generate **synthetic real-estate photo enhancement pairs** (BEFORE/AFTER) using either:

- **Gemini image models** (aka “Nano Banana”) via `google-genai` (`GEMINI_API_KEY`)
- **OpenAI image models** via `openai` (`OPENAI_API_KEY`)

The generator is designed to create **new scenes** and then produce an **“AutoHDR/flambient-like” AFTER** image that preserves the scene (no staging/object changes) and only applies pro photo edits (exposure/WB/contrast/window pull/perspective, etc).

## Setup (virtual environment)

```bash
cd synthetic_data
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Or:

```bash
./setup_venv.sh
source .venv/bin/activate
```

### API keys

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
```

For Gemini, `google-genai` also accepts `GOOGLE_API_KEY` (it takes precedence if both are set).

Optional: create a repo-root `.env` file (it is already gitignored) with those values.

## Quick start

### 1) Generate 10 pairs (Gemini / Nano Banana)

```bash
python re_pairgen.py generate \
  --provider gemini \
  --num 10 \
  --out runs/gemini_test \
  --pipeline base_to_both \
  --gemini-model gemini-2.5-flash-image \
  --aspect 16:9 \
  --capture-style amateur
```

### 2) Generate 20 pairs (OpenAI `gpt-image-1`)

```bash
python re_pairgen.py generate \
  --provider openai \
  --num 20 \
  --out runs/openai_test \
  --pipeline before_to_after \
  --openai-model gpt-image-1 \
  --openai-size 1536x1024 \
  --capture-style amateur
```

### 3) Generate ~300 pairs (adjustable)

```bash
python re_pairgen.py generate \
  --provider gemini \
  --num 300 \
  --out runs/gemini_300pairs \
  --pipeline before_to_after \
  --gemini-model gemini-2.5-flash-image \
  --aspect 16:9 \
  --continue-on-error \
  --max-failures 50
```

## Using your existing dataset as style reference (optional)

If you have `../autohdr-real-estate-577/` in this repo, you can pass it as a *style reference* for the AFTER edit. The generator will randomly sample a reference BEFORE/AFTER pair and use it to better match your dataset’s “edit delta” (without recreating the same scene).

```bash
python re_pairgen.py generate \
  --provider gemini \
  --num 50 \
  --out runs/gemini_with_refs \
  --pipeline before_to_after \
  --ref-dataset ../autohdr-real-estate-577 \
  --ref-pairs 1
```

## Output format

Each run writes:

- `images/{id}_src.jpg` (BEFORE)
- `images/{id}_tar.jpg` (AFTER)
- `train.jsonl` with `{"src": "...", "tar": "...", ...metadata }` entries
- `run_meta.json` with the run configuration

This format is compatible with `nafnet-realestate/nafnet_realestate_pipeline/scripts/prepare_from_jsonl.py`.

## Safety / dataset alignment

The AFTER prompts are intentionally strict to minimize “content drift”:

- no object additions/removals
- no virtual staging
- only tonal/color/optical corrections
- window pull allowed

If you see drift, try:

- `--pipeline base_to_both` (best alignment)
- `--ref-dataset ... --ref-pairs 1`
- switch Gemini model to `gemini-3-pro-image-preview` (higher fidelity; optionally add `--gemini-image-size 2K`)

If the BEFORE/AFTER gap is too subtle, increase the degradation range:

- `--severity-min 4 --severity-max 5`

## Resume

If a run stops mid-way, re-run the same command with `--resume` and it will continue from the highest `id` already written in `train.jsonl`.
