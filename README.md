# JMangaTranslator-Fast v1

English | [中文](README.zh.md)

JMangaTranslator-Fast translates Japanese manga speech bubbles into Simplified Chinese, one bubble at a time. It is a 368M-parameter encoder–decoder model built for low latency: 8.5 ms per bubble on an RTX 3060 and 20.2 ms on an Apple M1 Pro. It runs on NVIDIA GPUs, Apple silicon and plain CPUs.

Architecture, training, evaluation and backend details are in the [technical report](TECHNICAL_REPORT.md).

## Quality

COMET-22 on 3,817 Manga109-s bubbles. The set keeps every bubble on which an OCR model made a mistake, so it is harder than average. "Ground truth" is the annotated text; "manga-ocr" is the same bubbles as read by manga-ocr. Reference translations were produced by Claude.

| System | Parameters | Ground truth | manga-ocr |
|---|---|---|---|
| **JMangaTranslator-Fast v1** | 368M | **0.8879** | **0.8315** |
| GalTransl-v4-4B | 4B | 0.8617 | 0.8022 |
| NanoSakura-2.2-0.2B | 0.2B | 0.8456 | 0.7827 |
| Sakura-1.5B-Qwen2.5-v1.0 | 1.5B | 0.8392 | 0.7787 |
| Hy-MT2-1.8B-JP-Manga-Finetune-zh-Hans-v1 | 1.8B | 0.8349 | 0.7850 |
| Hy-MT2-1.8B | 1.8B | 0.8299 | 0.7818 |
| opus-mt-ja-zh | 77M | 0.6849 | 0.6532 |
| M2M100-418M | 418M | 0.6582 | 0.6278 |
| NLLB-200-distilled-600M | 600M | 0.6122 | 0.5904 |

v1 is ahead of every other system on both inputs, and every paired 95% bootstrap interval excludes zero. Outside manga bubbles a larger model can still be better: on Murasaki light-novel paragraphs, translated sentence by sentence, v1 scores 0.8314 and GalTransl-v4-4B 0.8504.

## Speed

Median (p50) latency per bubble at batch size 1, from tokenization to the decoded string:

| Hardware | Backend | p50 |
|---|---|---|
| RTX 4090 | CUDA Graphs, fp16 | 4.7 ms |
| RTX 3060 12 GB | CUDA Graphs, fp16 | 8.5 ms |
| Tesla P4 | CUDA Graphs, fp16 | 30.0 ms |
| Apple M1 Pro | MLX, fp16, GPU | 20.2 ms |
| Apple M1 Pro | Core ML, fp16, Neural Engine | 32.4 ms |
| Apple M1 Pro | ONNX Runtime, fp32, CPU, 6 threads | 52.1 ms |
| AMD EPYC 9654 | ONNX Runtime, fp32, CPU, 16 threads | 51.8 ms |

For comparison, GalTransl-v4-4B (llama.cpp, Q6_K) takes 147.6 ms per bubble on the same RTX 3060.

## Usage

The full release (weights, exported models and code) is on [Hugging Face](https://huggingface.co/muscgab/JMangaTranslator-Fast) and [ModelScope](https://modelscope.cn/models/muscgab/JMangaTranslator-Fast). [GitHub](https://github.com/muscgab/JMangaTranslator-Fast) holds the code only.

```bash
hf download muscgab/JMangaTranslator-Fast --local-dir JMangaTranslator-Fast
# or: modelscope download --model muscgab/JMangaTranslator-Fast --local_dir JMangaTranslator-Fast
cd JMangaTranslator-Fast
```

Install the requirements for your hardware, then translate:

| Hardware | Requirements | Backend |
|---|---|---|
| NVIDIA GPU | `requirements-torch.txt` | `cuda-graphs` |
| Apple silicon | `requirements-mlx.txt` | `mlx` |
| Apple Neural Engine (macOS 15+) | `requirements-coreml.txt` | `coreml` |
| Any CPU | `requirements-onnx.txt` | `onnx` |

```bash
pip install -r requirements-onnx.txt
python translate.py "堪忍袋の緒が切れた！"
python translate.py < bubbles.txt > translations.txt
```

```python
from jmt_fast import load
tr = load("path/to/JMangaTranslator-Fast")
print(tr.translate("堪忍袋の緒が切れた！"))   # 忍无可忍了！
```

Give one bubble per line, with line breaks inside a bubble removed. By default the backend is chosen from what is installed: `cuda-graphs` on an NVIDIA GPU, `mlx` on Apple silicon with MLX, otherwise `onnx`. The Neural Engine needs `--backend coreml` (in Python, `load(..., backend="coreml")`).

## Limitations

- **Long text.** The model was trained on bubble-length input. Split paragraphs into sentences first; given whole paragraphs, it fails to stop normally on 64 of 100.
- **No context.** Each bubble is translated on its own, so names, pronouns and tone may differ between bubbles.
- **OCR errors.** Recognition errors cost more quality (0.8879 → 0.8315 with manga-ocr) than v1's lead over GalTransl.
- **Machine-made training targets.** No human translations were used in training; idioms, cultural references and wordplay remain well below human quality.

## Acknowledgments

- **OpenSakura.** This model depends heavily on OpenSakura data. The encoder's domain-adaptive pretraining used [OpenSakura-DS-260220-LN-ja-zh-PT-Adam](https://huggingface.co/datasets/OpenSakura/OpenSakura-DS-260220-LN-ja-zh-PT-Adam), the light-novel part of translation training used [OpenSakura-DS-260220-LN-ja-zh-ALIGNED-Eve](https://huggingface.co/datasets/OpenSakura/OpenSakura-DS-260220-LN-ja-zh-ALIGNED-Eve), and the decoder's Chinese tokenizer and embeddings come from an earlier model trained on OpenSakura data.
- **Deep encoder, shallow decoder.** The architecture follows the central idea of Kasai et al., [Deep Encoder, Shallow Decoder: Reevaluating Non-autoregressive Machine Translation](https://arxiv.org/abs/2006.10369) (ICLR 2021): a deep encoder paired with a very shallow autoregressive decoder keeps translation quality while making batch-size-1 decoding fast. This model uses a 25-layer encoder and a 2-layer decoder.
- **SB Intuitions** for [ModernBERT-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m), which initializes the encoder.
- **[manga-ocr](https://github.com/kha-white/manga-ocr)**, whose error statistics shaped the noise generator and whose output forms one of the evaluation conditions.
- **Manga109-s** and **Murasaki**, used for evaluation.

## License

- Model weights and exported models: CC BY-NC-SA 4.0 (`LICENSE-weights.md`). Attribution is required, commercial use is not permitted, and adapted models must be shared under the same license.
- Code: MIT (`LICENSE`).

Training used OpenSakura data (license "other", intended for research and model development) and the author's private manga text. Manga109-s and Murasaki were used for evaluation only. No weights or outputs of the other systems in the comparison are included.
