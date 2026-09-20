# QLoRA Hands-On: 4-bit Quantization + LoRA on OPT-1.3B

A hands-on measurement of **QLoRA** — combining 4-bit NF4 quantization with LoRA — to see the actual memory savings this technique provides, not just the theoretical claim.

This builds directly on an earlier LoRA/DoRA fine-tuning project (DistilBERT/IMDb sentiment classification). That project used a small (67M param) model where quantization wouldn't show a meaningful difference. This one deliberately uses a bigger model (`facebook/opt-1.3b`, 1.3B params) specifically to make the memory mechanics visible and measurable.

## What's inside

- `qlora_handson_walkthrough.ipynb` — a cell-by-cell notebook covering:
  - Loading a model in full precision (bf16) and measuring GPU memory
  - Loading the same model in 4-bit (NF4) via `BitsAndBytesConfig` and measuring the reduction
  - Applying LoRA on top of the quantized backbone
  - Fine-tuning on a small instruction-style dataset (Alpaca subset)
  - Measuring peak memory during actual training

## Why QLoRA

Plain LoRA freezes the pretrained backbone and trains only small low-rank correction matrices — but the frozen backbone itself still has to be loaded into memory at full precision, which becomes the real bottleneck at large model sizes (7B+ parameters).

QLoRA solves this by quantizing the frozen backbone to 4-bit (NF4 — a format tuned to how neural network weights are actually distributed, denser near zero where most weights cluster). The frozen 4-bit weights are only *read* during the forward pass (dequantized on-the-fly for the matrix multiply, then discarded), never updated — so precision loss there is tolerable. The LoRA adapters stay in bf16, since they're the only parameters actually receiving gradients and need continuous, high-precision values for gradient descent to work at all.

## Results

### Memory footprint, measured end-to-end

| Stage | GPU memory |
|---|---|
| Full precision (bf16) load, no training | 2.45 GB |
| 4-bit (NF4) load, no training | 0.78 GB |
| 4-bit + LoRA adapters added, no training | 0.99 GB |
| Peak memory during actual training | 1.99 GB |

**Quantization reduction:** 2.45 GB → 0.78 GB = **~3.14x smaller**, just from compressing the frozen backbone's storage format. (Theoretical ceiling is 4x, going from 16-bit to 4-bit; the real-world ratio is lower due to some layers — layer norms, embeddings — being kept at higher precision, plus quantization-constant overhead and fixed CUDA context memory.)

**LoRA addition cost:** adding trainable adapters on top of the 4-bit backbone only cost **~0.21 GB** (0.78 GB → 0.99 GB) — a small fraction of the frozen model's own footprint, for a 1.3B-parameter model.

**Training overhead:** peak memory roughly doubled once training started (0.99 GB → 1.99 GB), from activations, gradients, and Adam optimizer states for the *trainable* LoRA parameters only — none of this touches the frozen 4-bit weights.

**Comparison to full fine-tuning:** a full fine-tune of this same 1.3B model (all weights trainable, no quantization) would need roughly 2.45 GB (weights) + ~2.45 GB (gradients) + ~4.9 GB (Adam's momentum + variance) ≈ **~10 GB minimum**. QLoRA did it in under 2 GB peak — a gap that widens further as model size scales into the 7B/13B/70B range, which is why this combination is a standard approach for fine-tuning large models on limited hardware.

### Training run details

- Model: `facebook/opt-1.3b`
- Dataset: 300-example subset of `tatsu-lab/alpaca`, formatted as instruction/response pairs
- LoRA config: r=8, alpha=16, target modules `q_proj`, `v_proj`
- 1 epoch, training loss at step 10: 2.97 (early in training — this run was designed to demonstrate memory mechanics, not produce a fully instruction-tuned model)

## Setup

```bash
pip install transformers peft accelerate bitsandbytes datasets
```

Requires a CUDA GPU (`bitsandbytes` 4-bit quantization does not run on CPU). Run `qlora_handson_walkthrough.ipynb` top to bottom in Google Colab.

## Key takeaways

- 4-bit NF4 quantization delivers a real, measured ~3x memory reduction on the frozen backbone — close to, but below, the theoretical 4x ceiling from bit-width alone.
- LoRA adapters add negligible memory on top of a quantized backbone, since they're a small fraction of the model's total parameters.
- The bulk of *additional* memory during training comes from the parts that must stay unquantized: activations, gradients, and optimizer states for the trainable LoRA parameters — not from the frozen weights.
- This combination (4-bit frozen backbone + bf16 LoRA adapters) is what makes fine-tuning large (7B+) models feasible on a single consumer or free-tier GPU.

## Related

See the companion [LoRA/DoRA fine-tuning project](https://github.com/KishoreKumar477/DistilBERT-peft-imdb) (DistilBERT/IMDb sentiment classification) for the underlying LoRA/DoRA theory and a smaller-scale comparison.
