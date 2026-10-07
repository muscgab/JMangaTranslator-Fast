# JMangaTranslator-Fast v1

[English](README.md) | 中文

JMangaTranslator-Fast 把日文漫画对话框翻译成简体中文，一次翻译一个框。它是一个 368M 参数的编码器-解码器模型，专为低延迟设计：单框延迟在 RTX 3060 上为 8.5 ms，在 Apple M1 Pro 上为 20.2 ms。可在 NVIDIA 显卡、Apple 芯片和普通 CPU 上运行。

结构、训练、评测与各推理后端的细节见[技术报告](TECHNICAL_REPORT.zh.md)。

## 效果

在 Manga109-s 的 3,817 个对话框上测 COMET-22。测试集收入了 OCR 模型识别出错的全部框，因此比一般情况更难。"原文标注"为人工标注的原文，"manga-ocr"为同一批框经 manga-ocr 识别后的文本。参考译文由 Claude 翻译。

| 系统 | 参数量 | 原文标注 | manga-ocr |
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

两种输入下 v1 都领先其余所有系统，配对 bootstrap 的 95% 区间均不含 0。漫画对话框以外，更大的模型仍可能更好：在 Murasaki 轻小说段落上（逐句翻译），v1 为 0.8314，GalTransl-v4-4B 为 0.8504。

## 速度

batch=1 时单个对话框从分词到输出译文的延迟中位数（p50）：

| 硬件 | 后端 | p50 |
|---|---|---|
| RTX 4090 | CUDA Graphs，fp16 | 4.7 ms |
| RTX 3060 12 GB | CUDA Graphs，fp16 | 8.5 ms |
| Tesla P4 | CUDA Graphs，fp16 | 30.0 ms |
| Apple M1 Pro | MLX，fp16，GPU | 20.2 ms |
| Apple M1 Pro | Core ML，fp16，神经网络引擎 | 32.4 ms |
| Apple M1 Pro | ONNX Runtime，fp32，CPU 6 线程 | 52.1 ms |
| AMD EPYC 9654 | ONNX Runtime，fp32，CPU 16 线程 | 51.8 ms |

作为对比，GalTransl-v4-4B（llama.cpp，Q6_K）在同一张 RTX 3060 上每框需 147.6 ms。

## 使用

不想本地安装的话，可以直接在 Colab 或 Kaggle 里打开 notebook（加载一次，之后反复翻译）：

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muscgab/JMangaTranslator-Fast/blob/main/notebook.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/muscgab/JMangaTranslator-Fast/blob/main/notebook.ipynb)

完整发布包（权重、导出模型和代码）在 [Hugging Face](https://huggingface.co/muscgab/JMangaTranslator-Fast) 和 [ModelScope](https://modelscope.cn/models/muscgab/JMangaTranslator-Fast)，[GitHub](https://github.com/muscgab/JMangaTranslator-Fast) 只放代码。

```bash
hf download muscgab/JMangaTranslator-Fast --local-dir JMangaTranslator-Fast
# 或：modelscope download --model muscgab/JMangaTranslator-Fast --local_dir JMangaTranslator-Fast
cd JMangaTranslator-Fast
```

按硬件安装依赖，然后翻译：

| 硬件 | 依赖 | 后端 |
|---|---|---|
| NVIDIA 显卡 | `requirements-torch.txt` | `cuda-graphs` |
| Apple 芯片 | `requirements-mlx.txt` | `mlx` |
| Apple 神经网络引擎（macOS 15 及以上） | `requirements-coreml.txt` | `coreml` |
| 任意 CPU | `requirements-onnx.txt` | `onnx` |

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

每行输入一个对话框，框内换行需先去掉。默认按已安装的依赖自动选后端：有 NVIDIA 显卡用 `cuda-graphs`，Apple 芯片装了 MLX 用 `mlx`，否则用 `onnx`。神经网络引擎需指定 `--backend coreml`（Python 中为 `load(..., backend="coreml")`）。

## 局限

- **长文本**：模型按对话框长度的输入训练。整段文字请先按句切分；直接输入整段时，100 段中有 64 段无法正常结束。
- **无上下文**：每个框独立翻译，跨框的人名、代词和语气可能不一致。
- **OCR 误差**：识别错误造成的损失（换成 manga-ocr 输入后 0.8879 → 0.8315）大于 v1 相对 GalTransl 的领先幅度。
- **训练译文均为机器生成**：训练未使用人工译文，惯用语、文化指涉和双关与人工翻译仍有明显差距。

## 致谢

- **OpenSakura**：本模型在很大程度上依赖 OpenSakura 的数据。编码器的领域继续预训练使用 [OpenSakura-DS-260220-LN-ja-zh-PT-Adam](https://huggingface.co/datasets/OpenSakura/OpenSakura-DS-260220-LN-ja-zh-PT-Adam)，翻译训练中的轻小说部分使用 [OpenSakura-DS-260220-LN-ja-zh-ALIGNED-Eve](https://huggingface.co/datasets/OpenSakura/OpenSakura-DS-260220-LN-ja-zh-ALIGNED-Eve)，解码器的中文分词器和词嵌入来自此前用 OpenSakura 数据训练的模型。
- **深编码器、浅解码器**：结构沿用 Kasai 等人 [Deep Encoder, Shallow Decoder: Reevaluating Non-autoregressive Machine Translation](https://arxiv.org/abs/2006.10369)（ICLR 2021）的核心思路：深编码器搭配极浅的自回归解码器，既保住翻译质量，又让 batch=1 解码足够快。本模型为 25 层编码器、2 层解码器。
- **SB Intuitions** 的 [ModernBERT-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m)，用于初始化编码器。
- **[manga-ocr](https://github.com/kha-white/manga-ocr)**：噪声生成器按它的识别错误统计拟合，它的输出也是评测条件之一。
- **Manga109-s** 和 **Murasaki**：用于评测。

## 许可

- 模型权重与导出模型：CC BY-NC-SA 4.0（`LICENSE-weights.md`）。须署名，禁止商用，改编后的模型须以相同许可发布。
- 代码：MIT（`LICENSE`）。

训练使用了 OpenSakura 数据（许可为 other，面向研究与模型开发）和作者的私有漫画文本。Manga109-s 与 Murasaki 仅用于评测。本项目不包含对比中其他系统的权重或输出。
