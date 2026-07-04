---
layout: post
title: "The Pretraining Loop, From Scratch"
description: Before you need tensor parallelism or ZeRO, you need a working training loop. This article builds one from first principles — the minimal version, what it leaves out, and why the structure matters when you scale.
category: Pretraining Concepts · Part 1 of 10
date: 2026-06-11
read_time: 12 min read
---

# The Pretraining Loop, From Scratch

Pretraining a language model is fundamentally a loop: get a batch of tokens,
compute the loss, backpropagate, update the parameters, repeat. Every technique
in this series — mixed precision, gradient accumulation, distributed parallelism,
checkpointing — is in service of running that loop more efficiently, on more
data, and without it breaking.

This article builds the loop from scratch, identifies where the simplest version
breaks down, and shows how the structure needs to change to accommodate scale.

---

**This series**: ten articles building a pretraining stack from first principles.

| Article | Topic |
|---|---|
| **Part 1** (this article) | The training loop: gradient accumulation, mixed precision, checkpointing |
| **Part 2** | Token shards, memory-mapped access, DP-aware data streaming |
| **Part 3** | Distributed primitives: ranks, process groups, and collectives |
| **Part 4** | Data parallelism: gradient all-reduce and bucketed async DDP |
| **Part 5** | ZeRO optimizer sharding: cutting memory by sharding optimizer state and weights |
| **Part 6** | Tensor and sequence parallelism: sharding weight matrices across GPUs |
| **Part 7** | Pipeline parallelism: splitting model depth across nodes with microbatch schedules |
| **Part 8** | Context parallelism: training on very long sequences with ring attention |
| **Part 9** | Mixture of Experts and expert parallelism: scaling parameters without scaling compute |
| **Part 10** | Putting it all together: choosing and combining parallelism strategies |

**Prerequisites**: Python, PyTorch basics (`nn.Module`, `torch.optim`, tensor operations), and a working understanding of how gradient descent trains a neural network. No prior distributed training knowledge is assumed — Parts 3–9 introduce distributed concepts as needed.

---

## The Minimal Training Loop

Start with the smallest correct version:

```python
model = GPT(vocab_size=32000, n_layers=6, d_model=384).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for step in range(max_steps):
    batch = get_next_batch()  # {"input_ids": Tensor[B, T], "labels": Tensor[B, T]}
    loss = model(batch["input_ids"], batch["labels"])  # cross-entropy loss, returned as a scalar
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()  # zero AFTER step is fine; zeroing before backward also works.
    print(f"step={step}  loss={loss.item():.4f}")
```

This is a real training loop. A small language model trained with this code on
real data will learn. The loss will decrease. But it has several problems that
become significant at any real scale.

---

## Problem 1: Memory Limits Batch Size

The effective batch size for LLM pretraining is typically in the millions of
tokens per optimizer step. A GPT-3-scale training run uses a batch size of
roughly 3 million tokens per step.

A single forward pass on one GPU can only fit as many tokens as fit in memory.
For a 40GB A100, that might be 512 tokens × 32 sequences = 16,384 tokens per
step. To reach 3M tokens per step you would need either 183 A100s running DDP,
or gradient accumulation.

**Gradient accumulation** runs multiple forward/backward passes before each
optimizer step, accumulating gradients:

```python
optimizer.zero_grad()
for micro_step in range(grad_accum_steps):
    batch = get_next_batch()
    loss = model(batch["input_ids"], batch["labels"]) / grad_accum_steps  # scale before backward
    loss.backward()
# gradients have accumulated across all micro-steps
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

The division by `grad_accum_steps` is not optional. Without it, the sum of
gradients across micro-steps is `grad_accum_steps` times larger than the gradient
from a single full-batch forward pass. The optimizer step would take an
`grad_accum_steps`-times larger update, which is equivalent to multiplying the
learning rate by that factor.

With the division, the gradient from 8 micro-steps of batch size 16 is
numerically equivalent to one forward pass with batch size 128. This is the
intended behavior: gradient accumulation should be transparent to the optimizer.

---

## Problem 2: Float32 Is Expensive

Modern GPUs execute matrix multiplications roughly twice as fast in `bfloat16`
as in `float32`. The difference between training a 7B-parameter model in 10 days
and 20 days is often just precision.

**Mixed precision** uses lower precision for the compute-intensive forward and
backward passes while maintaining higher-precision optimizer state:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model(batch["input_ids"], batch["labels"]) / grad_accum_steps
loss.backward()
```

The autocast context manager handles the dtype conversions automatically. Inside
the context, operations that benefit from lower precision (linear layers,
attention matmuls) run in `bfloat16`. Operations that need full precision — layer
norm, softmax, loss computation, and reduction operations — stay in `float32`.
PyTorch maintains an internal allowlist that controls which operations cast; you
can check it in the [PyTorch autocast docs](https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16).
Casting `float32 → bfloat16` loses about 8 bits of mantissa precision, which is
acceptable for matrix multiplications (the dominant cost) but not for numerically
sensitive operations like normalization.

**`bfloat16` vs `float16`**. For LLM training, `bfloat16` is usually the better
default:

- `bfloat16` has the same 8-bit exponent range as `float32`, so it can represent
  the same magnitude of numbers without overflow
- `float16` has only a 5-bit exponent and overflows at values above ~65,500 —
  common in attention logits and gradient norms during LLM training
- `float16` training usually requires a `GradScaler`, which multiplies the loss
  before backward to avoid gradient underflow and then unscales before the
  optimizer step; `bfloat16` usually does not require one

So in practice, use `bfloat16` on hardware that supports it, such as A100/H100.
On older hardware such as V100 that does not support `bfloat16`, you usually
fall back to `float16`, which means adding a gradient scaler:

```python
# float16 path — use only on hardware that doesn't support bfloat16
scaler = torch.amp.GradScaler()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(batch["input_ids"], batch["labels"]) / grad_accum_steps
scaler.scale(loss).backward()
if micro_step == grad_accum_steps - 1:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

The `unscale_` call before gradient clipping is required — the scaler inflates
gradient norms to prevent underflow, and clipping against inflated norms would
cut too early.

---

## Problem 3: The Data Loop Is Wrong

The naive `get_next_batch()` abstraction hides a mismatch between how pretraining
data is structured and how supervised learning data is structured.

In supervised learning, a dataset is a finite set of labeled examples. You
shuffle them, iterate through all of them once per epoch, then repeat.

In pretraining, the data is usually treated as a very long token stream —
billions or trillions of tokens — and training examples are cut from that stream
as contiguous fixed-length windows. Most runs are budgeted by total tokens or
optimizer steps, not by epochs, and many do not complete a full pass through
the corpus. In practice, pipelines may shuffle at the document or shard level,
but batch construction on each worker is usually modeled as advancing a cursor
through a sequential token stream rather than sampling independent windows at
random.

The data pipeline for pretraining is a **stateful cursor** over a sequential
token stream:

```python
class PretrainingDataLoader:
    def __init__(self, dataset, seq_len, micro_batch_size):
        self.shard_idx = 0
        self.token_offset = 0

    def next_batch(self) -> dict[str, torch.Tensor]:
        # read seq_len + 1 tokens, return input_ids and labels (shifted by 1)
        ...
```

The `+1` in `seq_len + 1` deserves explanation. Each training example predicts
the next token at each position. An example with sequence length `T` consumes
`T` input tokens and `T` label tokens. The labels are the inputs shifted left by
one position:

```
input_ids:  [t_0, t_1, t_2, ..., t_{T-1}]
labels:     [t_1, t_2, t_3, ..., t_T    ]
```

So each window consumes `T + 1` raw tokens from the stream. This affects every
calculation: batch size in tokens, DP stride, initial rank offset.

The article [How We Store and Stream Tokens](how-we-store-and-stream-tokens.html)
covers the data pipeline in detail. For now, the key point is that `next_batch()`
is not a stateless function — it advances a cursor, and that cursor state must be
saved in checkpoints.

---

## Problem 4: Nothing Survives a Crash

A pretraining run that trains for 8,000 steps and crashes at step 8,001 starts
from scratch unless it has been checkpointing. Checkpointing is not an
optimization; it is required infrastructure.

A correct resume needs more than the model weights. It needs:

- **Model state**: parameters and buffers
- **Optimizer state**: Adam's first and second moment estimates, step count
- **Scheduler state**: current learning rate position in the schedule
- **Data cursor**: current shard and token offset
- **RNG state**: CPU and CUDA random states, to reproduce stochastic operations

Model-only checkpoints are a common mistake. The observable effects:

- **Missing optimizer state**: Adam's moment estimates (`m`, `v`) warm up over
  thousands of steps and encode information about per-parameter gradient variance.
  Discarding them means the first several thousand steps after resume look like a
  warm-up restart — high loss, slow convergence — even though the model weights
  are well-trained. The loss will recover, but you've wasted compute.
- **Missing scheduler state**: the learning rate schedule resumes from its saved
  position, not from step 0. But without optimizer state, the effective update size
  is wrong — the Adam moments are at zero, effectively making early post-resume
  steps behave as if the LR were much higher. The loss curve may look stable while
  the effective gradient updates are noisier than they should be.

These effects are subtle enough that a model-only checkpoint often "works" in the
sense that loss continues decreasing. But the run diverges from what an
uninterrupted run would have produced, and the model quality at any given step
count is lower.

Data cursor checkpointing is subtler. A run that resumes without restoring the
data cursor will re-read tokens the model has already seen, or read from a
different position than where training left off. The loss may continue
decreasing, which makes the problem invisible in the training plot. Over many
steps, the model trained on mis-ordered data will diverge from the correctly
resumed run.

A minimal checkpoint save:

```python
def save_checkpoint(step, model, optimizer, scheduler, dataloader, path):
    torch.save({
        "step": step,  # the step that just completed; resume will start from step+1
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "dataloader": dataloader.state_dict(),  # shard_idx, token_offset, consumed_tokens
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
    }, path)
```

And the corresponding resume:

```python
def load_checkpoint(path, model, optimizer, scheduler, dataloader):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    dataloader.load_state_dict(checkpoint["dataloader"])
    torch.set_rng_state(checkpoint["rng_cpu"])
    torch.cuda.set_rng_state(checkpoint["rng_cuda"])
    return checkpoint["step"] + 1  # resume from the step AFTER the checkpoint
```

---

## Problem 5: No Visibility

The minimal loop logs one number: the loss. That's not enough to tell whether
a run is healthy.

Useful metrics for a pretraining run:

| Metric | Why it matters |
|---|---|
| **Loss** | Primary training signal; expect noise step to step, but the trend should move down |
| **Learning rate** | Verifies warmup/decay scheduling and helps catch resume or config mistakes |
| **Gradient norm** | Large spikes can indicate instability; sudden collapse can indicate vanishing or missing gradients |
| **Tokens per second** | Throughput sanity check; drops can reveal dataloader stalls or synchronization overhead |
| **TFLOPS/GPU** | Rough efficiency proxy; most useful when comparing runs with similar model size and sequence length |
| **Reserved memory** | Tracks allocator behavior; unexpected upward drift can indicate retained tensors or fragmentation |
| **Loss scale** (fp16 only) | Numerical stability signal for `fp16`; repeated scale drops or skipped optimizer steps usually mean overflow is happening |

Logging all of these at every step is expensive. Log every 10–50 steps; use
window averaging to smooth the values between log points.

---

## The Full Single-GPU Loop

<div class="article-figure">
  <img src="../assets/training-loop-flow.svg" alt="Training loop flow: gradient accumulation, mixed precision, logging, and checkpointing">
</div>

---

Putting the above together:

```python
model = GPT(...).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
# CosineAnnealingLR decays the LR from the initial value to near-zero following
# a cosine curve over T_max steps. This is the standard LLM pretraining schedule:
# the LR stays high during most of training and tapers off at the end.
# Production runs often add a short linear warmup (100–2000 steps) before the
# cosine decay to avoid large updates at initialization; that's omitted here for brevity.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
dataloader = PretrainingDataLoader(dataset, seq_len=1024, micro_batch_size=4)

# Resume if a checkpoint exists
start_step = 0
if resume_from:
    start_step = load_checkpoint(resume_from, model, optimizer, scheduler, dataloader)

for step in range(start_step, max_steps):
    # gradient accumulation loop
    optimizer.zero_grad()
    for _ in range(grad_accum_steps):
        batch = dataloader.next_batch()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(batch["input_ids"], batch["labels"]) / grad_accum_steps
        loss.backward()

    # clip and step
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    # Note: scheduler.step() advances the LR schedule, so get_last_lr() below
    # returns the LR for the NEXT step, not the one just used. To log the LR
    # that was used for the current step, read it before scheduler.step().

    # log
    if step % log_every == 0:
        # batch_tokens = grad_accum_steps * micro_batch_size * seq_len
        # step_time: measure with time.perf_counter() around the accumulation loop
        # loss.item() here is the last micro-step's (scaled) loss; for logging,
        # accumulate a running sum across micro-steps and divide by grad_accum_steps.
        tokens_per_sec = batch_tokens / step_time
        print(f"step={step}  loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
              f"  grad_norm={grad_norm:.3f}  tok/s={tokens_per_sec:.0f}")

    # checkpoint
    if step % checkpoint_every == 0:
        save_checkpoint(step, model, optimizer, scheduler, dataloader, f"ckpt_{step}.pt")
```

This is a real, runnable pretraining loop. A 40M-parameter model on a single
4090 training on FineWeb-Edu data produces a loss curve that descends smoothly
from ~2.5 to below 2.0 in a few hundred steps. The infrastructure works.

---

## Where Single-GPU Training Breaks Down

Two limits motivate distributed training:

**Memory**. The parameters, gradients, and optimizer states for a 7B-parameter
model exceed the memory of any single GPU. In a common Adam mixed-precision
setup, each parameter typically requires:

- 2 bytes (model parameter, in bf16)
- 4 bytes (fp32 master parameter used by the optimizer)
- 4 bytes (optimizer first moment, in fp32)
- 4 bytes (optimizer second moment, in fp32)
- 2 bytes (gradient, in bf16)

That is about 16 bytes per parameter in a common bf16 Adam setup, or roughly
112 GB for a 7B-parameter model before activations. Depending on the exact
optimizer/runtime, the number can vary a bit, but the conclusion does not: a
7B model does not fit comfortably on a single 80 GB GPU for training.

**Throughput**. A rough compute-budget sanity check leads to the same
conclusion. Training a GPT-3-scale run requires on the order of `3.14 × 10²³`
floating point operations. An H100's dense bf16 peak is ~989 TFLOPS, and real
training workloads often sustain only 35–45% of peak once memory stalls, kernel
overheads, and other inefficiencies are included — call it ~400 TFLOPS of
useful throughput. Even at that optimistic rate, a single H100 would need about
25 years of nonstop training time. That is why large-scale pretraining is
distributed across many GPUs.

> **Tip**
> A common back-of-the-envelope estimate for dense Transformer training is
> `6ND`, where `N` is parameter count and `D` is the number of training tokens.
> The `6` comes from roughly `2N` FLOPs per token for the forward pass and `4N`
> more for the backward pass: about `2N` for propagating gradients through the
> activations and another `2N` for computing parameter gradients. For GPT-3, `N = 175B` and
> `D = 300B`, giving about `6 × 175e9 × 300e9 ≈ 3.14 × 10²³` FLOPs.

---

## Why the Trainer Structure Matters

So far this article has stayed on a single GPU. But the reason to keep the loop
small and explicit is that the next steps in the series will start replacing
pieces of it with distributed behavior.

Consider what changes once you add ZeRO. The optimizer is no longer a standard
AdamW over all parameters. ZeRO shards optimizer state across DP ranks, so each
rank steps only a subset of parameters. The `optimizer.step()` call is no longer
just a local update; it now has distributed semantics.

The same pattern appears elsewhere. Add DDP and gradient reduction enters the
loop. Add TP and the model itself is transformed. Add PP and the simple
accumulation loop turns into a pipeline schedule. If all of that logic lives
directly inside `fit()`, the training loop fills up with strategy-specific
branches like `if ddp`, `if zero3`, and `if pp`.

At that point, the trainer is no longer just a trainer. It has quietly become a
distributed runtime.

That is the design lesson to carry forward from this article:

- **Trainer**: decides how many steps to run, when to log, and when to checkpoint.
- **Runtime**: executes the step, owns the optimizer update, and coordinates phase boundaries.
- **Distributed features**: own behavior such as gradient sync, sharding, and pipeline scheduling.

Keeping those boundaries clean is what lets the same high-level training loop
survive as the system grows from single-GPU training to DDP, TP, PP, and ZeRO.
The later articles in this series are really about swapping in those distributed
behaviors without rewriting the trainer from scratch.

---

## What's Next in This Series

The next article covers the data pipeline in detail: how token shards are stored,
how memory-mapped access works, and how the data cursor is made deterministic and
resumable across restarts and distributed configurations.

After that, the series adds a short distributed-primitives primer before moving
into data parallelism, tensor and sequence parallelism, and ZeRO optimizer
sharding — each building on the single-GPU loop established here.
