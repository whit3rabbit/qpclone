# qpclone

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/whit3rabbit/qpclone/blob/main/Qwen3_TTS_Voice_Cloning_to_PIPER.ipynb)

Clone any voice from a 3-5 second audio sample into a fully offline Piper TTS model.

This is a single Google Colab notebook that chains Qwen3-TTS (voice cloning) into Piper (lightweight VITS) to produce a deployable `.onnx` model you can run on a Raspberry Pi, Home Assistant, or any device that supports [Piper TTS](https://github.com/rhasspy/piper).

## How it works

1. Upload a short voice clip (3-5 seconds, clean speech, one speaker)
2. Qwen3-TTS generates ~1000 synthetic training samples in that voice
3. A pretrained Piper VITS model is fine-tuned on those samples
4. You get an ONNX model file that speaks in the cloned voice -- completely offline

## Requirements

| What | Details |
|------|---------|
| Voice sample | Clean 3-5 second WAV or MP3. No background noise, single speaker. |
| Hugging Face token | Free account at [huggingface.co](https://huggingface.co/join). Create a token (Read access) at [Settings > Tokens](https://huggingface.co/settings/tokens). Add it as a Colab secret named `HF_TOKEN` (click the key icon in the sidebar, toggle notebook access on). |
| Google Drive | ~2-4 GB free space for project files |
| Colab GPU | T4 GPU (free tier works). `Runtime > Change runtime type > T4 GPU` |
| Time | ~6 hours total (2-3 hrs generation + 2-3 hrs training) |

## Quick start

1. Open the notebook in Google Colab
2. Edit the settings cells at the top (project name, language, sample count)
3. Run the cells in order -- upload your voice clip when prompted
4. Walk away. Everything after the upload is automated.
5. Find your `.onnx` model on Google Drive when it finishes

## Configuration

All parameters are set via Colab form fields in the first few cells. The defaults are tuned for a T4 GPU (16 GB VRAM) on the free Colab tier.

### Project settings (Cell 1)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PROJECT_NAME` | -- | Name for your voice project (becomes a folder on Drive) |
| `LANGUAGE` | `English` | Target language. Determines which pretrained checkpoint is downloaded. |
| `REF_AUDIO_MODE` | `upload_if_missing` | `upload_if_missing` prompts only on first run. `always_upload` re-prompts every run. `use_existing_only` skips upload. |
| `REF_TEXT` | _(blank)_ | Transcript of your reference audio. Leave blank for automatic transcription via Qwen3-ASR. |

### Qwen3-TTS generation settings (Cells 8, 10)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `QWEN_MODEL_ID` | `Qwen3-TTS-12Hz-1.7B-Base` | 1.7B or 0.6B | 1.7B produces higher quality but needs ~8 GB VRAM. 0.6B needs ~4 GB. |
| `DTYPE` | `bfloat16` | bf16/fp16/fp32 | Inference precision for Qwen. bfloat16 is the native training dtype and the best default. |
| `NUM_SAMPLES` | 1000 | 100-3000 | Number of synthetic training clips to generate. 500-2000 recommended for fine-tuning. |
| `TEXT_DATASET_ID` | `MikhailT/lj-speech` | any HF dataset | Source of transcripts for generation. LJ Speech has ~13k English sentences. |
| `MAX_TEXT_CHARS` | 150 | 50-500 | Max characters per transcript sent to Qwen. See [why 150](#why-max_text_chars-150). |
| `MIN_TEXT_CHARS` | 10 | 1-50 | Minimum characters. Filters out trivially short sentences. |
| `MAX_DURATION_S` | 10.0 | any | Generated clips longer than this are discarded. See [why 10s](#why-max_duration_s-10). |
| `OUTPUT_SAMPLE_RATE` | 22050 | 16000/22050/44100 | Piper expects 22050 Hz mono. Qwen outputs at 24kHz and is resampled. |

### Piper training settings (Cells 9, 10)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `PIPER_MAX_EPOCHS` | 4000 | 100-10000 | Total fine-tuning epochs. See [why 4000](#why-4000-epochs). |
| `PIPER_BATCH_SIZE` | 8 | 4-64 | Samples per gradient step. See [why 8](#why-batch-size-8). |
| `PIPER_PRECISION` | 32 | 16 or 32 | Training precision. fp32 is the only tested configuration for VITS GAN training. fp16 is available but untested. |
| `PIPER_CHECKPOINT_EVERY` | 10 | 5-100 | Save a checkpoint every N epochs. |
| `PIPER_VALIDATION_SPLIT` | 0.05 | 0.01-0.2 | Fraction of data held out for validation. |
| `PIPER_RESUME` | `True` | bool | Auto-resume from last checkpoint on disconnect/restart. |
| `PIPER_PRETRAINED_CHECKPOINT` | `en/en_US/lessac/medium` | path or `none` | Pretrained model for fine-tuning. Set to `none` to train from scratch (needs 5000+ samples). |

### Pretrained checkpoints

Fine-tuning from a pretrained Piper checkpoint produces much better results with small datasets compared to training from scratch. The notebook auto-downloads the right checkpoint for your language from [rhasspy/piper-checkpoints](https://huggingface.co/rhasspy/piper-checkpoints).

Supported languages with checkpoints: English, Chinese, German, French, Russian, Portuguese, Spanish, Italian.

Japanese and Korean fall back to the English checkpoint (cross-language fine-tuning -- results may vary).

### Audio processing

Generated audio is:
- Resampled from 24kHz (Qwen output) to 22050Hz (Piper input)
- Peak-normalized to 0.95 to prevent clipping
- Saved as 16-bit PCM WAV (required by Piper's mel spectrogram processing)
- Clips under 0.2s or over `MAX_DURATION_S` are discarded

## Technical notes

### Why `MAX_TEXT_CHARS` 150

LJ Speech sentences average ~45 characters. The previous default of 250 let Qwen generate audio clips up to 15 seconds long. Piper's VITS training pads every sample in a batch to the length of the longest spectrogram in that batch. A single 15-second clip forces 7 other samples to be zero-padded to the same length, multiplying memory usage across the entire batch. 150 characters keeps most generated clips in the 3-8 second range, which is closer to the LJ Speech distribution (most clips 2-10s) and avoids these padding spikes.

### Why `MAX_DURATION_S` 10

`--max-phoneme-ids 300` caps text (phoneme) length but does not cap audio length. Short text can still produce long audio because Qwen's generation speed varies with prosody and pauses. The 10-second cap is a safety net that catches the outliers `MAX_TEXT_CHARS` misses. 10 seconds covers 99%+ of LJ Speech clips while keeping peak spectrogram memory within T4/L4 limits at batch size 8.

### Why batch size 8

Piper docs recommend batch size 32 on 24 GB GPUs. On a T4 (16 GB), real Colab sessions have 1-2 GB consumed by background processes and the Colab kernel itself. Batch 12 works on a clean GPU but OOMs on real sessions. Batch 8 is the reliable T4 configuration. On an L4 (24 GB) or A100, you can increase this -- batch size mainly affects training speed, not model quality ([Piper #703](https://github.com/rhasspy/piper/issues/703), [#189](https://github.com/rhasspy/piper/discussions/189)).

### Why 4000 epochs

Community guides and the Piper docs suggest ~1000 epochs for fine-tuning. However, with synthetic data (which has less speaker variation than real recordings), more epochs help the model converge on the target voice. 4000 is a reasonable upper bound for 1000 samples with fine-tuning -- checkpoints are saved every 10 epochs, so you can stop early or pick the best checkpoint. Training from scratch needs far more (10,000+).

### Why fp32 training precision

All official Piper documentation, community guides, and the rhasspy/piper training scripts use fp32. VITS uses adversarial (GAN) training where the generator and discriminator losses need full-precision gradients to remain stable. fp16 is exposed as an option but is untested for this architecture and may cause training instability or NaN losses.

### GPU memory management

The notebook never loads two large models simultaneously. Qwen3-ASR is loaded and unloaded before Qwen3-TTS loads. Qwen3-TTS is unloaded before Piper training starts. The training subprocess is launched with inline environment variables (`JAX_PLATFORMS=cpu`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, etc.) to prevent transitive dependencies (JAX, TensorFlow) from allocating GPU memory. Optimizer states are stripped from pretrained checkpoints before fine-tuning, saving ~8 GB of VRAM that Adam momentum buffers would otherwise consume.

## Pipeline

<p align="center">
  <img src="assets/flow.svg" alt="Voice cloning pipeline" width="700">
</p>

## Resume support

All progress is saved to `project.json` on Google Drive. If Colab disconnects:

- **Generation**: Picks up where it left off (skips already-generated samples)
- **Training**: Resumes from the last checkpoint automatically

## Project structure on Google Drive

```
Piper_Qwen_Projects/{project}/
  project.json          # persistent state
  reference/            # uploaded voice samples
  dataset/
    wavs/               # generated WAV training files
    metadata.csv        # LJSpeech format metadata
  piper/
    piper/              # cloned rhasspy/piper repo
    pretrained/         # downloaded pretrained checkpoint
  piper_training/
    training_runs/      # timestamped training output dirs
```

## Dependencies

Installed automatically by the notebook:

- [qwen-tts](https://github.com/QwenLM/Qwen3-TTS) -- voice cloning model
- [piper](https://github.com/rhasspy/piper) -- lightweight VITS TTS (archived Oct 2025, with compatibility fixes applied)
- pytorch-lightning 1.9.x -- pinned for Piper compatibility
- piper-phonemize (or piper-phonemize-cross for Python 3.12+)
- espeak-ng, sox -- system packages for phonemization and audio

## License

See individual model licenses:
- Qwen3-TTS: [Qwen License](https://github.com/QwenLM/Qwen3-TTS)
- Piper: [MIT License](https://github.com/rhasspy/piper)
