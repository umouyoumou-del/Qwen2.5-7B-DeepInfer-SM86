---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: deepinfer-sm86
tags:
  - qwen2.5
  - deepinfer
  - sm86
  - rtx-3050
  - w4a16
  - w3a16
  - int4
  - int3
  - gqa
  - cuda-graph
---

# Qwen2.5-7B-DeepInfer-SM86

Qwen2.5-7B-Instruct 的 **DeepInfer SM86 混合 W4/W3** 权重与运行代码。

目标设备：

```text
NVIDIA RTX 3050 Laptop GPU
4 GB VRAM
SM86
Windows
```

这是 **DeepInfer model_executor 级** 的 7B 混合精度部署包，不是完整
DeepInfer engine。

- 父项目：<https://github.com/umouyoumou-del/DeepInfer-SM86>
- 3B 版本：<https://github.com/umouyoumou-del/Qwen2.5-3B-DeepInfer-SM86>
- 权重 Release：<https://github.com/umouyoumou-del/Qwen2.5-7B-DeepInfer-SM86/releases/tag/v0.1.0>
- 权重格式：DeepInfer layered mixed W4/W3 checkpoint
- checkpoint 大小：约 **2.930 GiB**

---

## 本次发布（v0.1.0）

- 发布 Qwen2.5-7B-Instruct 的 DeepInfer SM86 混合 W4/W3 权重；
- attention / embedding 4-bit，MLP / lm_head 3-bit；
- 包含 28 层分层 checkpoint、embedding、3-bit lm_head、tokenizer、manifest；
- 包含构造期 fused qkv / gate-up 实现；
- 包含 W4A16 GEMV、W3A16 GEMV、128/256-slot GQA kernel 源码；
- 包含 OpenAI 兼容 7B 网关、量化脚本和基准结果；
- 权重作为四个 Release 分片上传，不进入普通 Git 历史。

---

## 结果速览

4 GB RTX 3050 隔离环境实测：

| 指标 | 结果 |
|---|---:|
| checkpoint | 2.930 GiB |
| 服务加载 | 4.445 秒 |
| prefill | 3.061 秒 |
| decode 平均 | **25.962 tokens/s** |
| decode 中位 | **26.188 tokens/s** |
| 中位单 token | 38.186 ms |
| 峰值 allocated | 3.196 GiB |
| 峰值 reserved | 3.357 GiB |
| 测速 context | 64 |
| 生成后 allocated | 2.970 GiB |

基准配置：

```text
prompt tokens = 30
max_new_tokens = 24
max_cache = 64
warmup_replays = 3
```

由于 4 GB 边界，7B 更适合短上下文。网关支持到 256，但在有其它 GPU
程序占用时会进入 WDDM paging；4 GB 上推荐先用 `--max-cache 64`。

---

## 与 3B / Ollama 的关系

| 后端 | decode | 一次请求 | context |
|---|---:|---:|---:|
| 3B DeepInfer SM86 | 44.777 t/s | 0.9009 s | 256 |
| **7B DeepInfer SM86 mixed** | **25.962 / 26.188 t/s** | 约 3.1 s prefill + decode | 64 测速配置 |
| Ollama 3B Q4_K_M | 57.285 t/s | 0.4637 s | 4096 |

7B 质量更高，但速度更慢，且 4 GB 上不能和 3B 同时常驻。

---

## 权重下载

权重 **不放在普通 Git 文件里**，因为 checkpoint 约 2.930 GiB。使用 Release 的
四个分片：

```text
Qwen7B-Mixed-SM86.tar.part-00   1000 MiB
Qwen7B-Mixed-SM86.tar.part-01   1000 MiB
Qwen7B-Mixed-SM86.tar.part-02   1000 MiB
Qwen7B-Mixed-SM86.tar.part-03     约 12 MiB
```

下载：

```powershell
gh release download v0.1.0 `
  --repo umouyoumou-del/Qwen2.5-7B-DeepInfer-SM86 `
  --dir weights
```

合并：

```powershell
cd weights
cmd /c copy /b Qwen7B-Mixed-SM86.tar.part-00+Qwen7B-Mixed-SM86.tar.part-01+Qwen7B-Mixed-SM86.tar.part-02+Qwen7B-Mixed-SM86.tar.part-03 Qwen7B-Mixed-SM86.tar
```

解压：

```powershell
tar -xf Qwen7B-Mixed-SM86.tar -C D:\Qwen7B-Mixed-SM86
```

校验：

```powershell
sha256sum -c SHA256SUMS
```

当前资产校验值：

```text
3903c6d982a79069eb320c95ec7e13add8008e9c156f840559e7503eb639322d  Qwen7B-Mixed-SM86.tar.part-00
25e7d117236848ca3cbac02073f7fcf1a96bcd35455474f0c25fde080f8b3ba2  Qwen7B-Mixed-SM86.tar.part-01
e7d37d861460db741c933d1eab7a9fe017aaf14a8b5552b309df7818f0bf6d9b  Qwen7B-Mixed-SM86.tar.part-02
1508dec300ce3ee474b4ba3404ad7258fd27bc8637c59c99c18cc25da5ec7e0c  Qwen7B-Mixed-SM86.tar.part-03
ec9a5baf33ebba0b83c94028a3d58fc1ad5381e28c9e8e949ac3c1dfc744fe2e  Qwen7B-Mixed-SM86.tar
```

解压后目录应包含：

```text
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
manifest.json
embed.safetensors
lm_head3.safetensors
norm.safetensors
layer_00.safetensors
...
layer_27.safetensors
```

---

## 快速启动

### 1. 环境

已验证：

```text
Python 3.10
torch 2.5.1+cu121
CUDA 12.1
CuPy 14.2.0
safetensors
transformers
```

```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
pip install -r requirements-sm86.txt
```

### 2. 启动 7B 网关

```powershell
cd D:\Qwen2.5-7B-DeepInfer-SM86
python apps\server.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --backend mixed `
  --model-id deepinfer-sm86-7b-mixed `
  --max-cache 64 `
  --port 18000
```

接口：

```text
GET  http://127.0.0.1:18000/health
GET  http://127.0.0.1:18000/v1/models
POST http://127.0.0.1:18000/v1/chat/completions
```

OpenAI 客户端 Base URL：

```text
http://127.0.0.1:18000/v1
```

模型名：

```text
deepinfer-sm86-7b-mixed
```

### 3. 直接 loader

```python
from deepinfer.model_executor.models.qwen2_mixed_sm86 import (
    Qwen2MixedForCausalLM,
)

model = Qwen2MixedForCausalLM.from_checkpoint(
    r"D:\Qwen7B-Mixed-SM86",
    device="cuda",
    group_size=128,
)
```

---

## 量化格式

```text
attention q/k/v/o : 4-bit, group 128
embedding         : 4-bit, group 128
MLP gate/up/down  : 3-bit, group 128
lm_head           : 3-bit, group 128
```

3-bit 格式：

```text
8 个 signed 3-bit 值压进 3 bytes
量化范围 [-4, 3]
scale = max(abs(group)) / 3
```

有效位宽：

```text
约 3.30 bpw
```

checkpoint：

```text
2.930 GiB
```

这不是整体 Q4；整体 Q4 7B 是另一个约 3.658 GiB 的实验，在 4 GB 上会进入
paging，decode 约 7.9 tokens/s。

---

## 融合内核与执行路径

### W4A16

```text
symmetric int4
group size = 128
scale = max(abs(group)) / 7
```

- `kernels/qwen2_w4a16_gemv.cu`
- 一个 warp 处理一个输出 row；
- x 预加载到 shared memory；
- 单 token decode 使用 W4A16 GEMV；
- prefill 走分块 dequant + cuBLAS。

### W3A16

- `kernels/qwen2_w3a16_gemv.cu`
- 8 个 signed 3-bit 值压进 3 bytes；
- gate/up 使用 512 threads；
- down 使用 768 threads；
- 与显式反量化 matvec 的 cosine 为 1.0。

### 融合 qkv / gate-up

checkpoint 的分层文件保存的是：

```text
q_proj / k_proj / v_proj
gate_proj / up_proj
```

loader 在模型构造期直接装入：

```text
Qwen2Attention(fused_qkv=...)
Qwen2MLP(fused_gate_up=...)
```

不是先构造普通层再替换，避免 4 GB 边缘上的 allocator 碎片。

**说明：** 融合是模块级融合，复用 `qwen2_w4a16_gemv` / `qwen2_w3a16_gemv`
CUDA kernel；不存在另一个独立的 `fused_matmul.cu`。

### 256-slot GQA

- `kernels/qwen2_gqa_decode_256.cu`
- 一个 block 处理一个 query head；
- 最多 256 个静态 cache slot；
- 支持 28 Q / 4 KV heads（7B）；
- 静态 KV cache + CUDA Graph。

---

## 模型规格

```text
base model            Qwen2.5-7B-Instruct
architecture          Qwen2ForCausalLM
hidden_size           3584
intermediate_size     18944
num_hidden_layers     28
num_attention_heads   28
num_key_value_heads   4
head_dim              128
vocab_size            152064
tie_word_embeddings   false
quantization          attention/embed 4-bit, MLP/head 3-bit
```

权重内容：

```text
28 层混合 W4/W3 linear
4-bit embedding
3-bit lm_head
layernorm / manifest / tokenizer
```

---

## 目录

```text
README.md
LICENSE
NOTICE
requirements-sm86.txt

model/
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  manifest.json
  norm.safetensors

deepinfer/
  model_executor/models/qwen2.py
  model_executor/models/qwen2_w4a16.py
  model_executor/models/qwen2_mixed_sm86.py
  hugging_face/qwen2_config.py
  ...

apps/
  server.py           OpenAI 兼容网关
  runtime.py          常驻模型 + static cache + Graph
  ask.py              CLI

scripts/
  run_qwen7b_mixed_gpu_harness.py
  quantize_qwen7b_mixed.py
  finish_qwen7b_mixed_quantize.py
  quantize_qwen7b_head3.py
  quantize_qwen7b_q4.py
  bench_productivity_backends.py
  export_sm86_kernels.py

kernels/
  qwen2_w4a16_gemv.cu
  qwen2_w3a16_gemv.cu
  qwen2_gqa_decode.cu
  qwen2_gqa_decode_256.cu

docs/
  qwen7b_mixed_benchmark.json
  qwen7b_mixed_summary.json
  ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md
  PRODUCTIVITY_BENCHMARK_3B_7B_OLLAMA.md
```

---

## 已知限制

- 当前是 **DeepInfer model_executor 级**，`full_engine=false`；
- 没有完整 CacheEngine、Sampler、scheduler/worker、continuous batching；
- 4 GB 上 7B 非常接近显存边界；
- 后台 GPU 占用约 500 MiB 时，prefill 可能从 3 秒变成 53 秒以上；
- 测速配置是 64 context；256 context 在干净环境才建议尝试；
- 3-bit MLP / lm_head 是有损量化，输出不等于 BF16；
- 3B 与 7B 不能同时常驻 4 GB；
- GPU harness 在 Python 3.10 下验证，Python 3.12 package 验证未完成；
- 融合 qkv / gate-up 是构造期模块融合，不是独立 fused GEMM 文件。

---

## 许可

- 基础模型 Qwen2.5-7B-Instruct 遵循其原始许可；
- DeepInfer 派生代码遵循 Apache License 2.0；
- 详见 `LICENSE` 与 `NOTICE`。

---

## 相关项目

- 内核与量化工程：<https://github.com/umouyoumou-del/DeepInfer-SM86>
- 3B 版本：<https://github.com/umouyoumou-del/Qwen2.5-3B-DeepInfer-SM86>
- 上篇：父仓库 `docs/ARTICLE_1_DEEPINFER_SM86_KERNEL.md`
- 下篇：`docs/ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md`
- 3B/7B/Ollama 基准：`docs/PRODUCTIVITY_BENCHMARK_3B_7B_OLLAMA.md`
