# JMangaTranslator-Fast v1 Technical Report

English | [中文](TECHNICAL_REPORT.zh.md)

This report covers the architecture, training, evaluation and inference backends of JMangaTranslator-Fast v1. For an overview and quick start, see the [README](README.md).

## Architecture

The encoder is ModernBERT-ja-310m (25 layers, width 768). Its vocabulary was extended with 1,633 missing characters, and it was further pretrained on Japanese light-novel and manga text.

Hidden states from all 25 encoder layers are combined by a learned weighted fusion and passed through a SwiGLU bridge (hidden size 2,048), together with 2 null tokens, to the decoder. After training, the fusion weights concentrate on layers 15–23.

The decoder is a 2-layer autoregressive Transformer: width 768, 12 heads, SwiGLU FFN of 4,096, and a 22,935-entry Chinese vocabulary. Decoding is greedy.

The encoder runs once per bubble and the decoder runs once per output token, so compute is placed in the encoder and the decoder is kept shallow. In ablations, disabling either decoder FFN reduces COMET by 0.17–0.21.

The model cannot emit Japanese kana:
- training pairs with kana in the target were removed;
- the decoder vocabulary contains no kana-bearing pieces;
- byte sequences that would spell kana in UTF-8 are masked during generation.

## Training

All training ran on a single RTX 4090, about 15 GPU-hours in total.

**Domain-adaptive pretraining (about 9 hours).** The corpus is OpenSakura PT-Adam Japanese novels and manga text, sampled 9:1 with a sequence length of 1,024. Training ran for 15,260 steps over 1.0B token positions, covering the novels about 1.0 time and the manga text about 3.6 times. It had three phases:
- The first 500 steps trained only the embeddings of the added characters, at learning rate 1e-3 with no weight decay.
- The whole model was then trained on a WSD schedule: learning rate 1e-4, 500 warmup steps, then constant, with a 30% masking rate.
- Over the final 20% of steps the learning rate decayed to 0 following 1−√t, and the masking rate dropped to 15%.

Masked tokens are replaced with [MASK] 80% of the time, a random token 10% of the time, and left unchanged 10% of the time. The prediction head is computed only at masked positions.

**Translation training (about 6 hours, 54,000 steps).**
- The encoder was frozen for the first 2,000 steps. After that the whole model was trained, with the encoder learning rate at 0.1× the base rate.
- Learning rate 3e-4, 1,000 warmup steps, cosine decay to 10%.
- Weight decay 0.01, gradient clipping 1.0, label smoothing 0.1, dropout 0.1, bf16.
- In total, 37.8M rows and about 0.83B Chinese target tokens.

**Training data.** All training targets are model-generated:
- 70% of sampled rows come from OpenSakura light novels (22.7M rows), with Chinese produced by GLM-4.7.
- 30% come from OCR text of the author's private manga collection (2.95M rows; 300 chapters held out), translated by a large language model.

**Noise robustness.**
- Source text is perturbed by a noise generator fitted to real manga-ocr error statistics, applied to 25% of manga rows and 5% of novel rows.
- 36,327 pairs of real OCR output and target translations are added, oversampled 3×.

Sentence-final punctuation follows the source. No context is used in training, and all sources are bubble-length.

## Evaluation

COMET-22 (Unbabel/wmt22-comet-da) is the primary metric and chrF is secondary. All differences are reported with 95% intervals from 4,000 paired bootstrap resamples.

### Manga109-s

The test set is drawn from 20,000 Manga109-s bubbles. It contains every bubble on which manga-ocr or mangaOCR-NAR made at least one recognition error, 3,817 bubbles in total, so it is skewed toward hard OCR cases. Reference translations were produced by Claude. Manga109-s text is used for evaluation only.

| System | Parameters | Ground-truth text | manga-ocr output | mangaOCR-NAR output |
|---|---|---|---|---|
| JMangaTranslator-Fast v1 | 368M | 0.8879 | 0.8315 | 0.8408 |
| GalTransl-v4-4B | 4B | 0.8617 | 0.8022 | 0.8125 |
| NanoSakura-2.2-0.2B | 0.2B | 0.8456 | 0.7827 | 0.7868 |
| Sakura-1.5B-Qwen2.5-v1.0 | 1.5B | 0.8392 | 0.7787 | 0.7869 |
| Hy-MT2-1.8B-JP-Manga-Finetune-zh-Hans-v1 | 1.8B | 0.8349 | 0.7850 | 0.7939 |
| Hy-MT2-1.8B | 1.8B | 0.8299 | 0.7818 | 0.7899 |
| opus-mt-ja-zh | 77M | 0.6849 | 0.6532 | 0.6558 |
| M2M100-418M | 418M | 0.6582 | 0.6278 | 0.6367 |
| NLLB-200-distilled-600M | 600M | 0.6122 | 0.5904 | 0.5937 |

Each system runs as its model card describes, with no tuning on this set. GalTransl, Sakura-1.5B and the two Hy-MT2 models run on llama-server (llama.cpp 4f54067) with the GGUF files and sampling settings published by their authors: GalTransl Q6_K and Sakura-1.5B fp16, both with an empty glossary; Hy-MT2-1.8B Q8_0; and fumetodev's manga fine-tune of Hy-MT2-1.8B, Q4_K_M, without a terminology block. NanoSakura runs with its upstream code and greedy decoding. opus-mt-ja-zh (shun89), M2M100 and NLLB run in transformers with beam size 4.

Paired differences on the same input:

| Comparison | Ground truth | manga-ocr | mangaOCR-NAR |
|---|---|---|---|
| v1 − GalTransl-v4-4B | +0.0261 [0.0232, 0.0291] | +0.0293 [0.0263, 0.0325] | +0.0283 [0.0247, 0.0318] |
| v1 − NanoSakura-2.2-0.2B | +0.0423 [0.0388, 0.0459] | +0.0488 [0.0451, 0.0527] | +0.0540 [0.0501, 0.0580] |
| v1 − Sakura-1.5B-Qwen2.5-v1.0 | +0.0486 [0.0449, 0.0524] | +0.0527 [0.0488, 0.0569] | +0.0540 [0.0497, 0.0583] |
| v1 − Hy-MT2-1.8B-JP-Manga-Finetune-zh-Hans-v1 | +0.0530 [0.0496, 0.0562] | +0.0465 [0.0431, 0.0499] | +0.0470 [0.0434, 0.0506] |
| v1 − Hy-MT2-1.8B | +0.0580 [0.0548, 0.0613] | +0.0497 [0.0464, 0.0530] | +0.0509 [0.0473, 0.0546] |
| v1 − opus-mt-ja-zh | +0.2030 [0.1973, 0.2084] | +0.1783 [0.1728, 0.1835] | +0.1850 [0.1795, 0.1902] |
| v1 − M2M100-418M | +0.2297 [0.2238, 0.2353] | +0.2037 [0.1980, 0.2092] | +0.2041 [0.1983, 0.2096] |
| v1 − NLLB-200-distilled-600M | +0.2757 [0.2700, 0.2816] | +0.2411 [0.2350, 0.2472] | +0.2471 [0.2412, 0.2530] |

An anonymized blind preference test was also run on 100 bubbles: 50 with ground-truth input and 50 with manga-ocr input, a single rater, references hidden, ties allowed. Score shares were v1 42.3% [35.7, 49.0], GalTransl 24.3% and NanoSakura 18.3%.

### Murasaki

The set has 200 light-novel paragraphs. Each paragraph is split into sentences, translated sentence by sentence, and re-joined for scoring. Comparability with the official leaderboard has not been verified.

| System | COMET |
|---|---|
| GalTransl-v4-4B (whole paragraph) | 0.8570 |
| GalTransl-v4-4B (per sentence) | 0.8504 |
| JMangaTranslator-Fast v1 (per sentence) | 0.8314 |
| NanoSakura-2.2-0.2B (per sentence) | 0.8271 |

v1 − GalTransl (per sentence) is −0.0190 [−0.0218, −0.0164]; v1 − NanoSakura is +0.0043 [−0.0017, 0.0127].

### Speed

Latency is the wall time of one bubble at batch size 1, from tokenization to the decoded string, measured after 20 warm-up bubbles. Inputs are Manga109-s ground-truth text.

**NVIDIA GPUs.** 500 bubbles, p50 / p90 in ms, PyTorch 2.8.0 with CUDA 12.6. Host CPUs are a Xeon Gold 6430 (RTX 4090), a Xeon E5-2690 v3 (RTX 3060) and a Xeon E5-2683 v4 (Tesla P4).

| System | RTX 4090 | RTX 3060 12 GB | Tesla P4 |
|---|---|---|---|
| JMangaTranslator-Fast v1, CUDA Graphs fp16 | 4.7 / 6.9 | 8.5 / 13.0 | 30.0 / 41.9 |
| JMangaTranslator-Fast v1, PyTorch eager fp32 | 25.3 / 38.3 | 54.7 / 84.0 | 74.0 / 115.5 |
| NanoSakura-2.2-0.2B, PyTorch fp32 | 32.3 / 58.3 | 63.0 / 113.8 | 79.1 / 144.7 |
| GalTransl-v4-4B, llama.cpp Q6_K | 64.5 / 115.2 | 147.6 / 281.4 | 354.1 / 668.8 |

The CUDA Graphs version captures the encoder once per input-length bucket (32, 64, 128 tokens) and one decoder step as CUDA graphs. Kana masking, argmax and the KV-cache update run inside the step graph, and the host reads back one token per step. Weights and activations are fp16; the input of each normalization layer is divided by a per-layer power of two (1 where not needed) to stay within fp16 range. Its output is identical to the fp32 eager version on 98.8% of bubbles on the RTX 4090 and RTX 3060, and on 98.6% on the Tesla P4.

NanoSakura runs with its upstream modeling code and greedy decoding. GalTransl runs on llama-server (llama.cpp 4f54067, one slot, all layers on the GPU) with temperature 0.3 and top_p 0.8. On the RTX 3060, GalTransl ran at the card's 170 W power limit for most of its run.

**Apple M1 Pro.** 200 bubbles, p50 / p90 in ms. "Loaded" means 10 other processes were saturating the CPU. The last column is the share of bubbles with output identical to the fp32 PyTorch version.

| Backend | Idle | Loaded | Identical |
|---|---|---|---|
| MLX fp16, GPU | 20.2 / 29.4 | 22.2 / 32.9 | 98.5% |
| Core ML fp16, ANE (all ops on ANE) | 32.4 / 56.3 | 37.8 / 69.6 | 99% |
| Core ML fp16, GPU | 43.2 / 67.9 | 59.6 / 100.1 | 100% |
| ONNX Runtime fp32, CPU, 6 threads | 52.1 / 88.8 | 103.8 / 178.2 | 100% |

The ANE inference process has a peak memory footprint of 690.7 MB and uses 8.4 ms of CPU time per bubble. The first load, including compilation, takes 47.1 s. On the full Manga109-s set, the Core ML fp16 and PyTorch versions score 0.8878 and 0.8879 COMET respectively, a difference of −0.0000 [−0.0003, 0.0002].

**x86 CPU.** With ONNX Runtime fp32 on an AMD EPYC 9654 (container limited to 32 cores), 200 bubbles, p50 is 51.8 ms with 16 threads, 79.4 ms with 6 threads and 142.0 ms with 2 threads. Outputs are identical to the fp32 PyTorch version.

## Limitations

**Long input.** Training sources were at most about 115 tokens long. When whole paragraphs are given directly, COMET is 0.5956, and 64 of 100 paragraphs fail to terminate normally. Long text should be split into sentences first.

**No context.** Bubbles are translated independently, so names, pronouns and register are not kept consistent across bubbles.

**OCR errors.** Switching from ground-truth text to manga-ocr output lowers COMET from 0.8879 to 0.8315. This drop is larger than v1's margin over GalTransl.

**Repetition and quality.** On ground-truth input over 3,817 bubbles, 97 suspected repetitions were detected automatically, including legitimate repeated exclamations. Because no human translations were used in training, idioms, cultural references and wordplay remain clearly below human quality.

## Inference backends

| Backend | Hardware | Requirements | Precision | Output identical to `torch` fp32 |
|---|---|---|---|---|
| `cuda-graphs` | NVIDIA GPU | `requirements-torch.txt` | fp16 | 98.8% (RTX 3060, 500 bubbles) |
| `mlx` | Apple silicon GPU | `requirements-mlx.txt` | fp16 | 98.5% (M1 Pro, 200 bubbles) |
| `coreml` | Apple Neural Engine, macOS 15+ | `requirements-coreml.txt` | fp16 | 97.0% (M1 Pro, 200 bubbles) |
| `onnx` | CPU | `requirements-onnx.txt` | fp32 | 100% (500 bubbles) |
| `torch` | CPU, CUDA or MPS | `requirements-torch.txt` | fp32 | reference |

`auto` selects `cuda-graphs` when an NVIDIA GPU is available, `mlx` on Apple silicon with MLX installed, and otherwise `onnx`. The `torch` backend accepts up to 256 source tokens; the other backends use fixed-shape graphs and keep the first 128 source tokens and at most 128 output tokens. Every backend reproduces the outputs of the measurement runs in the Speed section bubble for bubble (`tools/verify.py`).

| File | Size | Description |
|---|---|---|
| `model.safetensors` | 1.47 GB | All weights, fp32, encoder included |
| `config.json` | 0.4 KB | Decoder hyperparameters and special token ids |
| `tokenizer/` | 6.8 MB | Japanese tokenizer (ModernBERT-ja with added characters) and encoder architecture |
| `joint.model`, `vocab.json` | 0.9 MB | Chinese SentencePiece model and decoder vocabulary mapping |
| `norm_scales.json` | 5 KB | Normalization scales for the fp16 CUDA Graphs backend |
| `onnx/` | 1.54 GB | Encoder (variable length) and one-step decoder, fp32, with host-side embedding tables |
| `coreml/` | 772 MB | Encoder (lengths 32 / 64 / 128) and one-step decoder, fp16, with host-side embedding tables |
| `jmt_fast/`, `translate.py`, `tools/` | | Inference code, command-line tool, checkpoint conversion and verification |
| `src/` | | Training, export and measurement code of this release |
