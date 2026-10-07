# JMangaTranslator-Fast v1

[English](README.md) | 中文

JMangaTranslator-Fast v1 是一个日译中（简体）的漫画对话框翻译模型，参数量 367.5M，采用编码器-解码器结构。模型以单个对话框为翻译单位，面向 batch=1 的低延迟推理。单框延迟 p50 在 RTX 3060 上为 8.5 ms，在 Apple M1 Pro 上为 20.2 ms。

## 结构

编码器为 ModernBERT-ja-310m（25 层，宽 768）。在原词表上补充了 1,633 个缺字，并在日文轻小说与漫画文本上做了领域继续预训练。

编码器全部 25 层的隐状态经可学习加权融合后，通过 SwiGLU 桥接层（隐藏 2,048）送入解码器，另附 2 个 null token。训练后，融合权重主要分布在第 15–23 层。

解码器为 2 层自回归 Transformer：宽 768，12 头，SwiGLU FFN 4,096，中文词表 22,935。解码采用贪心策略。

编码器每框只计算一次，解码器逐 token 串行，因此计算量集中在编码器、解码器保持极浅。消融实验中，关闭任意一层解码器 FFN，COMET 下降 0.17–0.21。

训练数据中删除了译文含假名的样本，解码器词表不含任何含假名的 piece，生成时屏蔽可拼出假名的 UTF-8 字节序列，因此模型不会输出日文假名。

## 训练

全部训练在单张 RTX 4090 上完成，约 15 卡时。

**领域继续预训练（约 9 小时）。** 语料为 OpenSakura PT-Adam 日文小说与漫画文本，按 9:1 混合采样，序列长度 1,024。共 15,260 步、10 亿 token 位置，小说约 1.0 遍，漫画约 3.6 遍。训练分三个阶段：
- 前 500 步只训练新增字的嵌入，学习率 1e-3，不加 weight decay；
- 随后全模型训练，采用 WSD 调度：学习率 1e-4，warmup 500 步后保持恒定，遮盖率 30%；
- 最后 20% 的步数按 1−√t 衰减到 0，遮盖率同时降为 15%。

遮盖方式为 80% 替换成 [MASK]、10% 随机 token、10% 保持原样，只在被遮盖位置计算预测头。

**翻译训练（约 6 小时，共 54,000 步）。**
- 前 2,000 步冻结编码器，此后全模型训练，编码器学习率为整体的 0.1 倍。
- 学习率 3e-4，warmup 1,000 步，余弦衰减至 10%。
- weight decay 0.01，梯度裁剪 1.0，label smoothing 0.1，dropout 0.1，bf16。
- 累计训练 3,780 万行、约 8.3 亿中文 token。

**训练数据。** 训练数据全部为模型蒸馏译文：
- 70% 来自 OpenSakura 轻小说（2,270 万行），中文由 GLM-4.7 生成；
- 30% 来自作者私有漫画集的 OCR 文本（295 万行，留出 300 章不参与训练），中文由大模型翻译。

**抗噪训练。**
- 用按 manga-ocr 真实错误统计拟合的噪声生成器扰动原文，漫画行扰动 25%，小说行扰动 5%。
- 加入 36,327 行真实 OCR 识别结果与对应译文的配对，过采样 3 倍。

句尾标点保持与原文一致。训练不使用上下文，原文长度均在单个对话框范围内。

## 评测

主指标为 COMET-22（Unbabel/wmt22-comet-da），chrF 为辅；差异均以 4,000 次配对 bootstrap 给出 95% 区间。

### Manga109-s

测试集取自 Manga109-s 的两万个对话框：凡 manga-ocr 或 mangaOCR-NAR 至少一方识别出错的框全部收入，共 3,817 框，因此偏向 OCR 难例。参考译文由 Claude 翻译。Manga109-s 文本仅用于评测。

| 系统 | 参数量 | 原文标注 | manga-ocr 识别结果 | mangaOCR-NAR 识别结果 |
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

各系统均按其模型卡的方式运行，未针对本测试集调参。GalTransl、Sakura-1.5B 与两个 Hy-MT2 模型运行于 llama-server（llama.cpp 4f54067），使用作者发布的 GGUF 文件与采样设置：GalTransl 为 Q6_K、Sakura-1.5B 为 fp16，二者术语表均为空；Hy-MT2-1.8B 为 Q8_0；fumetodev 基于 Hy-MT2-1.8B 的漫画微调版为 Q4_K_M，不带术语块。NanoSakura 使用上游代码，贪心解码。opus-mt-ja-zh（shun89）、M2M100 与 NLLB 在 transformers 中以 beam 4 解码。

同一输入下的配对差：

| 对比 | 原文标注 | manga-ocr | mangaOCR-NAR |
|---|---|---|---|
| v1 − GalTransl-v4-4B | +0.0261 [0.0232, 0.0291] | +0.0293 [0.0263, 0.0325] | +0.0283 [0.0247, 0.0318] |
| v1 − NanoSakura-2.2-0.2B | +0.0423 [0.0388, 0.0459] | +0.0488 [0.0451, 0.0527] | +0.0540 [0.0501, 0.0580] |
| v1 − Sakura-1.5B-Qwen2.5-v1.0 | +0.0486 [0.0449, 0.0524] | +0.0527 [0.0488, 0.0569] | +0.0540 [0.0497, 0.0583] |
| v1 − Hy-MT2-1.8B-JP-Manga-Finetune-zh-Hans-v1 | +0.0530 [0.0496, 0.0562] | +0.0465 [0.0431, 0.0499] | +0.0470 [0.0434, 0.0506] |
| v1 − Hy-MT2-1.8B | +0.0580 [0.0548, 0.0613] | +0.0497 [0.0464, 0.0530] | +0.0509 [0.0473, 0.0546] |
| v1 − opus-mt-ja-zh | +0.2030 [0.1973, 0.2084] | +0.1783 [0.1728, 0.1835] | +0.1850 [0.1795, 0.1902] |
| v1 − M2M100-418M | +0.2297 [0.2238, 0.2353] | +0.2037 [0.1980, 0.2092] | +0.2041 [0.1983, 0.2096] |
| v1 − NLLB-200-distilled-600M | +0.2757 [0.2700, 0.2816] | +0.2411 [0.2350, 0.2472] | +0.2471 [0.2412, 0.2530] |

另做了 100 框的匿名盲评（原文标注与 manga-ocr 输入各 50 框，评审一人，参考译文隐藏，允许并列）。按得分占比：v1 42.3% [35.7, 49.0]，GalTransl 24.3%，NanoSakura 18.3%。

### Murasaki

200 个轻小说段落，按句切分后逐句翻译，再拼回整段评分。该流程与官方排行榜的可比性未经核实。

| 系统 | COMET |
|---|---|
| GalTransl-v4-4B（整段） | 0.8570 |
| GalTransl-v4-4B（逐句） | 0.8504 |
| JMangaTranslator-Fast v1（逐句） | 0.8314 |
| NanoSakura-2.2-0.2B（逐句） | 0.8271 |

v1 − GalTransl（逐句）为 −0.0190 [−0.0218, −0.0164]；v1 − NanoSakura 为 +0.0043 [−0.0017, 0.0127]。

### 速度

延迟为 batch=1 时单个对话框从分词到输出译文的耗时，先预热 20 框再计时。输入为 Manga109-s 原文标注。

**NVIDIA 显卡。** 500 框，p50 / p90（ms），PyTorch 2.8.0、CUDA 12.6。主机 CPU 分别为 Xeon Gold 6430（RTX 4090）、Xeon E5-2690 v3（RTX 3060）、Xeon E5-2683 v4（Tesla P4）。

| 系统 | RTX 4090 | RTX 3060 12 GB | Tesla P4 |
|---|---|---|---|
| JMangaTranslator-Fast v1，CUDA Graphs fp16 | 4.7 / 6.9 | 8.5 / 13.0 | 30.0 / 41.9 |
| JMangaTranslator-Fast v1，PyTorch eager fp32 | 25.3 / 38.3 | 54.7 / 84.0 | 74.0 / 115.5 |
| NanoSakura-2.2-0.2B，PyTorch fp32 | 32.3 / 58.3 | 63.0 / 113.8 | 79.1 / 144.7 |
| GalTransl-v4-4B，llama.cpp Q6_K | 64.5 / 115.2 | 147.6 / 281.4 | 354.1 / 668.8 |

CUDA Graphs 版将编码器按输入长度分档（32、64、128 token）各录制一张 CUDA 图，解码一步录制为一张图；假名屏蔽、argmax 与 KV 缓存更新均在图内完成，主机每步只读回一个 token。权重与激活均为 fp16，各归一化层的输入按层除以一个 2 的幂（不需要时为 1），以保持在 fp16 范围内。其译文与 fp32 eager 版逐框相同的比例，在 RTX 4090 与 RTX 3060 上为 98.8%，在 Tesla P4 上为 98.6%。

NanoSakura 使用上游模型代码，贪心解码。GalTransl 运行于 llama-server（llama.cpp 4f54067，单槽，全部层在 GPU），temperature 0.3，top_p 0.8。RTX 3060 上 GalTransl 运行期间大部分时间处于该卡 170 W 的功耗上限。

**Apple M1 Pro。** 200 框，p50 / p90（ms）。"满载"指另有 10 个进程占满 CPU。最后一列为与 fp32 PyTorch 版译文逐框相同的比例。

| 后端 | 空闲 | 满载 | 相同 |
|---|---|---|---|
| MLX fp16，GPU | 20.2 / 29.4 | 22.2 / 32.9 | 98.5% |
| Core ML fp16，ANE（全部算子在 ANE） | 32.4 / 56.3 | 37.8 / 69.6 | 99% |
| Core ML fp16，GPU | 43.2 / 67.9 | 59.6 / 100.1 | 100% |
| ONNX Runtime fp32，CPU 6 线程 | 52.1 / 88.8 | 103.8 / 178.2 | 100% |

ANE 推理进程的峰值内存为 690.7 MB，每框 CPU 时间 8.4 ms，首次加载（含编译）47.1 s。Core ML fp16 版与 PyTorch 版在 Manga109-s 全集上的 COMET 为 0.8878 vs 0.8879，差 −0.0000 [−0.0003, 0.0002]。

**x86 CPU。** 在 AMD EPYC 9654（容器限 32 核）上以 ONNX Runtime fp32 测试 200 框，p50 为：16 线程 51.8 ms，6 线程 79.4 ms，2 线程 142.0 ms。译文与 fp32 PyTorch 版逐框相同。

## 局限

**长文本。** 模型训练时原文长度不超过约 115 个 token。整段长文本直接输入时，COMET 为 0.5956，100 段中有 64 段未能正常结束，长文本应先按句切分。

**无上下文。** 各框独立翻译，跨框的人名、代词与语气不保证一致。

**OCR 误差。** 输入由原文标注换为 manga-ocr 识别结果时，COMET 由 0.8879 降至 0.8315，降幅大于 v1 相对 GalTransl 的领先幅度。

**重复与译文质量。** 3,817 框原文标注输入中，自动检测到疑似重复 97 例（含正常的感叹重复）。训练数据不含人工译文，惯用语、文化指涉和双关的处理与人工翻译仍有明显差距。

## 使用

完整发布包（权重、导出模型和代码）在 [Hugging Face](https://huggingface.co/muscgab/JMangaTranslator-Fast) 和 [ModelScope](https://modelscope.cn/models/muscgab/JMangaTranslator-Fast)，[GitHub](https://github.com/muscgab/JMangaTranslator-Fast) 只放代码。

```bash
hf download muscgab/JMangaTranslator-Fast --local-dir JMangaTranslator-Fast
# 或：modelscope download --model muscgab/JMangaTranslator-Fast --local_dir JMangaTranslator-Fast
cd JMangaTranslator-Fast
```

```bash
pip install -r requirements-onnx.txt        # 或 requirements-torch / -coreml / -mlx
python translate.py "堪忍袋の緒が切れた！"
python translate.py --backend onnx < bubbles.txt > translations.txt
```

每行输入为一个对话框，框内换行需先去除。`--model` 指向发布目录（默认为 `translate.py` 所在目录）。在 Python 中：

```python
from jmt_fast import load
tr = load("path/to/release", backend="auto")
print(tr.translate("堪忍袋の緒が切れた！"))   # 忍无可忍了！
```

| 后端 | 硬件 | 依赖 | 精度 | 与 `torch` fp32 逐框相同 |
|---|---|---|---|---|
| `cuda-graphs` | NVIDIA GPU | `requirements-torch.txt` | fp16 | 98.8%（RTX 3060，500 框） |
| `mlx` | Apple 芯片 GPU | `requirements-mlx.txt` | fp16 | 98.5%（M1 Pro，200 框） |
| `coreml` | Apple 神经网络引擎，macOS 15 及以上 | `requirements-coreml.txt` | fp16 | 97.0%（M1 Pro，200 框） |
| `onnx` | CPU | `requirements-onnx.txt` | fp32 | 100%（500 框） |
| `torch` | CPU、CUDA 或 MPS | `requirements-torch.txt` | fp32 | 基准 |

`auto` 在有 NVIDIA GPU 时选 `cuda-graphs`，在装有 MLX 的 Apple 芯片上选 `mlx`，否则选 `onnx`。`torch` 后端接受最多 256 个原文 token；其余后端使用固定形状的计算图，保留前 128 个原文 token，最多输出 128 个 token。各后端的译文与速度一节测速时的结果逐框一致（`tools/verify.py`）。

| 文件 | 大小 | 说明 |
|---|---|---|
| `model.safetensors` | 1.47 GB | 全部权重，fp32，含编码器 |
| `config.json` | 0.4 KB | 解码器超参数与特殊 token |
| `tokenizer/` | 6.8 MB | 日文分词器（ModernBERT-ja，含补字）与编码器结构 |
| `joint.model`, `vocab.json` | 0.9 MB | 中文 SentencePiece 模型与解码器词表映射 |
| `norm_scales.json` | 5 KB | fp16 CUDA Graphs 后端的归一化缩放系数 |
| `onnx/` | 1.54 GB | 编码器（长度可变）与单步解码器，fp32，含主机侧嵌入表 |
| `coreml/` | 772 MB | 编码器（长度 32 / 64 / 128）与单步解码器，fp16，含主机侧嵌入表 |
| `jmt_fast/`、`translate.py`、`tools/` | | 推理代码、命令行工具、权重转换与验证 |
| `src/` | | 本版本的训练、导出与测速代码 |

## 许可

模型权重与导出模型以 CC BY-NC-SA 4.0 发布（`LICENSE-weights.md`）：须署名，禁止商用，改编后的模型须以相同许可发布。源代码以 MIT 许可发布（`LICENSE`）。

| 来源 | 许可 | 用途 |
|---|---|---|
| sbintuitions/modernbert-ja-310m | MIT | 编码器初始权重 |
| OpenSakura 数据集 | other（面向研究与模型开发） | 继续预训练与翻译训练 |
| 作者私有漫画文本 | 不公开 | 翻译训练 |
| Manga109-s、Murasaki | 各自许可 | 仅评测 |

解码器的中文分词器与词嵌入，初始化自作者此前在 OpenSakura 数据上训练的模型。评测中的其他系统仅作对比，本项目不包含其权重或输出。
