# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

qpclone is a single Jupyter notebook (`Qwen3_TTS_Voice_Cloning_to_PIPER.ipynb`) that implements an end-to-end voice cloning pipeline. It takes a 3-5 second voice sample, uses Qwen3-TTS to generate synthetic training data, then fine-tunes a Piper VITS model to produce a deployable offline ONNX TTS model.

Designed to run in Google Colab with a T4 GPU. All persistent state lives on Google Drive.

## Pipeline Architecture

```
Reference Audio (3-5s WAV/MP3)
    |
    v
Qwen3-ASR auto-transcription (if REF_TEXT blank) -- load 0.6B, transcribe, unload
    |
    v
Qwen3-TTS voice feature extraction (xvector + ref audio + transcript)
    |
    v
Qwen3-TTS synthesis -- generates ~1400 WAV samples using DailyDialog conversational transcripts
    |
    v
Piper preprocessing -- WAVs + metadata.csv -> phoneme IDs + audio tensors (.pt)
    |
    v
Piper VITS training -- fine-tune for 4000 epochs (default)
    |
    v
ONNX export -> deployable offline TTS model
```

GPU memory is managed explicitly: Qwen3-ASR is loaded and unloaded (Cell 12) before Qwen3-TTS loads (Cell 13). Qwen3-TTS is unloaded (Cell 21) before Piper training starts. No models coexist in VRAM.

## Notebook Cell Map

The first 6 cells are the interactive setup (user changes project name, grants Drive access, uploads audio). Everything after Cell 7 is automated.

| Cell | Purpose |
|------|---------|
| 0 | Markdown: getting started guide + Colab badge |
| 1 | Voice clone settings (project name, language, ref audio mode) |
| 2 | Install dependencies (pip, apt) |
| 3 | Imports and utility functions |
| 4 | Mount Google Drive + create project paths |
| 5 | Init project state (`project.json`) -- minimal, only needs Cell 1 vars |
| 6 | Reference audio upload/reuse |
| 7 | Markdown: "walk away" divider |
| 8 | Dataset & generation settings (NUM_SAMPLES, text dataset, emotion diversity config) |
| 9 | Piper training settings (epochs, batch size, pretrained checkpoint) |
| 10 | Advanced settings (model size, device, precision, sample rate) |
| 11 | Save full config to project state |
| 12 | Auto-transcribe reference audio with Qwen3-ASR (skipped if REF_TEXT provided or already transcribed) |
| 13 | Load Qwen3-TTS model |
| 14 | Pre-compute voice clone prompt (speaker features extracted once) |
| 15 | Load text dataset (DailyDialog) from Hugging Face. Emotion-aware: detects emotion field, balances text selection across emotion categories (caps neutral at `NEUTRAL_RATIO`), builds parallel `texts`/`text_emotions` lists |
| 16 | Markdown: Piper section divider |
| 17 | Generate training samples (WAVs + metadata.csv). Audio is peak-normalized to 0.95 and saved as 16-bit PCM. Clips exceeding `MAX_DURATION_S` are discarded to prevent OOM during training. When `ENABLE_EMOTION_SAMPLING` is on, varies temperature/top_p/top_k per emotion via `EMOTION_SAMPLING_PARAMS`. Writes `emotion_metadata.csv` for reference (does not affect Piper's `metadata.csv`). |
| 18 | Clone Piper repo and fix dependencies for Python 3.12+ |
| 19 | Download pretrained Piper checkpoint from `rhasspy/piper-checkpoints` dataset repo on HuggingFace (auto-matches language). Must use `repo_type="dataset"` for HF Hub API calls. |
| 20 | Preprocess dataset for Piper training |
| 21 | Free GPU memory (unload Qwen3-TTS) |
| 22 | Train/resume Piper VITS model (3-tier checkpoint: resume > fine-tune > scratch) |
| 23 | Project status summary |
| 24 | Export to ONNX and test synthesis |

## Key Configuration Parameters

Defined in Cells 1 and 8-10 as Colab form fields:

- `PROJECT_NAME` / `LANGUAGE` / `DRIVE_BASE_DIR` -- project identity
- `NUM_SAMPLES` (100-3000, default 1400) -- synthetic training samples to generate
- `PIPER_MAX_EPOCHS` (100-10000, default 3000) -- training duration
- `PIPER_BATCH_SIZE` (4-64, default 24) -- auto-configured by `GPU_PRESET`
- `QWEN_MODEL_ID` -- 1.7B-Base (~8GB VRAM) or 0.6B-Base (~4GB VRAM)
- `OUTPUT_SAMPLE_RATE` -- Qwen outputs 24kHz, resampled to 22050Hz for Piper
- `TEXT_DATASET_ID` -- Hugging Face dataset for transcripts (default: `daily_dialog`)
- `MAX_TEXT_CHARS` (50-500, default 150) -- max characters per transcript sent to Qwen. Capped at 150 to keep generated audio in the 3-8s range and avoid OOM from long spectrograms during training. DailyDialog turns are typically short conversational utterances.
- `MAX_DURATION_S` (default 10.0) -- generated audio clips longer than this are discarded. Prevents batch-padding OOM in Piper training.
- `PIPER_PRETRAINED_CHECKPOINT` -- path within `rhasspy/piper-checkpoints` HF **dataset** repo (default: `en/en_US/lessac/medium`). Set to `none` to train from scratch.
- `EMOTION_BALANCED` (default True) -- balance text selection across DailyDialog emotion categories instead of sequential (83% neutral) selection
- `NEUTRAL_RATIO` (0.1-1.0, default 0.4) -- max fraction of neutral (emotion=0) texts when `EMOTION_BALANCED` is on
- `ENABLE_EMOTION_SAMPLING` (default True) -- vary temperature/top_p/top_k per emotion category during TTS generation. Parameters defined in `EMOTION_SAMPLING_PARAMS` dict in Cell 3 (hand-tunable).
- `EMOTION_FIELD` (default `"emotion"`) -- dataset column containing per-utterance emotion labels. Both features gracefully disable if the field is missing.

## Runtime Directory Structure (Google Drive)

```
/content/drive/MyDrive/Piper_Qwen_Projects/
  transcription_cache.json  -- global cross-project transcription cache (keyed by SHA256)
  {project_slug}/
  project.json          -- persistent state (tracks progress across sessions)
  reference/            -- uploaded voice samples
  dataset/
    wavs/               -- generated WAV training files
    metadata.csv        -- <basename>|<transcript> per line
    emotion_metadata.csv -- <basename>|<emotion_id>|<emotion_name>|<transcript> (reference only)
    config.json         -- generated by Piper preprocessing
    dataset.jsonl       -- generated by Piper preprocessing
  piper/
    piper/              -- cloned rhasspy/piper repo
    pretrained/         -- downloaded pretrained checkpoint (.ckpt) for fine-tuning
  piper_training/
    training_runs/      -- timestamped training output dirs
  piper_export/
    piper_model/        -- ONNX model output
    test_audio/         -- test synthesis output
```

## Key Dependencies

- `qwen-tts` -- Qwen3-TTS voice cloning model
- `qwen-asr` -- Qwen3-ASR automatic speech recognition (used for auto-transcribing reference audio)
- `piper-phonemize` (or `piper-phonemize-cross` for Python 3.12+) -- phoneme processing
- `pytorch-lightning==1.9.5` -- pinned for Piper compatibility (uses old import path)
- `espeak-ng` -- system package for phonemization
- `sox` -- system package for audio processing

## Notes for Editing

- The Piper repo (rhasspy/piper) was archived Oct 2025. Cell 18 contains specific dependency pins and compatibility shims for Python 3.12+ that are fragile -- understand them before modifying.
- `project.json` state file enables resume across Colab disconnects. Generation (Cell 17) and training (Cell 22) both support resumption via this state.
- Project state is initialized in two phases: Cell 5 creates the minimal state (before audio upload), Cell 11 saves full config (after all settings cells). This split allows the interactive cells to run before the user needs to touch config defaults.
- The `@title` and `@param` annotations are Colab form syntax -- they generate the UI widgets. Changing parameter names or types affects the Colab form rendering.
- Audio resampling from 24kHz (Qwen output) to 22050Hz (Piper input) happens in Cell 17 during generation. Audio is peak-normalized to 0.95 and written as 16-bit PCM. Mismatched sample rates or formats will produce poor training results.
- Cell 22 uses a 3-tier checkpoint priority: (1) resume interrupted run, (2) fine-tune from pretrained checkpoint, (3) train from scratch. Fine-tuning is the default first-run path and requires far fewer samples than training from scratch. For Priority 2, `optimizer_states` and `lr_schedulers` are set to empty lists in the pretrained checkpoint (saved as `.weights.ckpt`) and epoch/step are reset to 0. Keys must remain present (not removed) because Lightning 1.9.5 checks for key existence. The stripped checkpoint is re-prepared from the original every run (not cached) to ensure the fix applies to users with stale `.weights.ckpt` files. This reduces VRAM by ~8 GB (Adam momentum buffers) and makes `max_epochs` represent total fine-tuning epochs rather than an absolute count.
- Cell 12 auto-transcribes reference audio with Qwen3-ASR-0.6B when `REF_TEXT` is blank. The `ref_text_source` field in `state["ref_audio"]` tracks origin: `"manual"` (user typed it), `"asr"` (auto-transcribed), `"cache"` (resolved from global transcription cache), `"pending"` (not yet transcribed), or `"none"` (ASR returned empty). On resume, if `ref_text_source` is `"asr"` or `"cache"`, the saved transcript is reused without reloading ASR. Uploading new reference audio in Cell 6 resets `ref_text_source` to `"pending"`, forcing ASR to re-run (unless the global cache has an entry for that audio's SHA256).
- A global transcription cache at `{DRIVE_BASE_DIR}/transcription_cache.json` stores transcriptions keyed by SHA256 hash of the reference audio file. This allows cross-project reuse: same audio file = skip ASR regardless of project. Cell 6 checks the cache on upload, Cell 11 saves manual transcriptions to it, and Cell 12 both checks (second-chance) and saves ASR results to it. Helper functions `read_transcription_cache()` and `write_transcription_cache()` are defined in Cell 3.
- Qwen3-ASR-0.6B (~2-4GB VRAM in bfloat16) is fully unloaded (del + gc.collect + torch.cuda.empty_cache) before Qwen3-TTS loads. No model coexistence needed.
- Default training precision is fp32 (`PIPER_PRECISION = 32`), matching official Piper docs and all community guides. fp16 is untested for VITS GAN training but remains available in the dropdown. Default batch size is 24 (T4 preset). A `GPU_PRESET` dropdown auto-configures batch size and precision for common GPUs (T4=24, L4=32, A100-40=48, A100-80=64, V100=24, RTX 3090/4090=32). Training uses ~8 GB VRAM with `--max-phoneme-ids 300` capping per-batch peak memory.
- Training uses `--max-phoneme-ids 300` to drop sentences exceeding 300 phoneme IDs. This caps per-batch peak memory and prevents rare long sentences from causing OOM spikes (per Piper issue #703 and discussion #189). DailyDialog conversational turns are short and typically produce 30-100 phoneme IDs, so 300 drops almost nothing from the dataset while keeping peak memory well within T4 limits.
- `generate_voice_clone()` does not support style or emotion control. The `instruct` parameter only exists on `generate_custom_voice()` (built-in speakers) and `generate_voice_design()`. Any `style_instruction` kwarg passed to `generate_voice_clone()` is silently ignored. Emotion diversity is achieved indirectly via two levers: (1) curating emotionally diverse text from DailyDialog labels, and (2) varying sampling parameters (temperature, top_p, top_k) per emotion category to produce varied prosody.
- Cell 22 inlines env vars (`JAX_PLATFORMS=cpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `TF_FORCE_GPU_ALLOW_GROWTH=true`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) directly in the training command string so they always apply to the subprocess regardless of kernel state. The `os.environ` calls are kept as fallback for any imports that happen before the subprocess. `torch.set_float32_matmul_precision('medium')` is called before training to use Tensor Cores on L4/A100.
