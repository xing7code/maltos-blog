---
layout: post
title: "Pipeline Parallelism"
description: When a model is too large to fit on a node — or when tensor parallelism has hit its scaling limit — pipeline parallelism splits the model vertically across stages and keeps all GPUs busy with a schedule of microbatches. This article covers how the split works, the two schedules MALTOS implements, and where the bubble comes from.
category: Pretraining Concepts · Part 7 of 9
date: 2026-06-11
read_time: 14 min read
---

# Pipeline Parallelism

Tensor parallelism distributes each weight matrix across GPUs within a node.
With NVLink bandwidth, TP scales well to 4–8 GPUs per node. Scaling further —
across nodes — requires a different strategy. Pipeline parallelism (PP) splits
the model *vertically*: the first few transformer layers run on stage 0, the
next few on stage 1, and so on. Each stage holds only its assigned layers.

This solves a different memory problem than TP. TP reduces the memory per layer
(each GPU holds a fraction of each weight matrix). PP reduces the total number
of layers a GPU needs to hold. A 96-layer model split across 8 pipeline stages
means each GPU holds only 12 layers — reducing the parameter memory by 8×.

The cost is a new problem: **pipeline bubbles**. If stage 0 runs its forward
pass and waits for stage 7 to finish before doing anything else, most GPUs sit
idle most of the time. Eliminating that idle time is what pipeline schedules are
for.

---

## How the Model Is Split

MALTOS splits the model into three regions:

```python
class PipelineParallelSpec:
    head_layers: list[str]   # embedding, RoPE — only on stage 0
    pipe_layers: list[str]   # transformer block layers — split evenly across stages
    tail_layers: list[str]   # layer norm, lm_head — only on last stage
```

The transformer blocks are partitioned evenly. For a 24-layer model across
4 stages, each stage gets 6 layers. The plugin calls `_layer_range()`:

```python
def _layer_range(num_layers: int, stage_index: int, stage_count: int):
    base, remainder = divmod(num_layers, stage_count)
    start = stage_index * base + min(stage_index, remainder)
    width = base + (1 if stage_index < remainder else 0)
    return start, start + width
```

When layers don't divide evenly, the first `remainder` stages each get one extra
layer. For a 25-layer model across 4 stages: stage 0 gets 7 layers, stages 1–3
get 6 each. This front-loading is simply an artifact of the divmod arithmetic —
production systems often balance stages by *compute and memory* rather than raw
layer count, since stage 0 already carries the embedding and stage P-1 carries
the LM head (which for large vocabularies can outweigh several transformer
layers). MALTOS keeps the split simple.

Layers that don't belong to this stage are replaced with `_IdentityPipeLayer` —
a module whose `forward()` returns its first argument unchanged. This preserves
the module hierarchy (so `model.layers[i]` is still valid Python) without
executing or loading weights for those layers. Only the assigned layers have
real parameters; the others are structural placeholders.

---

## What Crosses Stage Boundaries

Only stage 0 receives raw token IDs as input. Every stage after that receives
a hidden state tensor — the output of the previous stage's last transformer
block — and never sees the original vocabulary tokens.

Each stage's output is a hidden state tensor — the activation after the last
transformer block on that stage. Stage `i` sends its output to stage `i+1`
using an async point-to-point send. "Async" here means the send is enqueued on
the CUDA stream and stage `i` can immediately start its next operation (e.g.,
the forward pass of the next microbatch) without waiting for the data to fully
arrive at stage `i+1`. Stage `i+1` calls `recv_work.wait()` when it actually
needs the data, which blocks until the transfer is complete:

```python
# Stage i: send activations to next stage (non-blocking)
send_buffer, send_work = self._send_activation_async(boundary_activation.detach())

# Stage i+1: receive activations from previous stage
input_activation, recv_work = self._recv_activation_async(micro_batch)
recv_work.wait()  # blocks until the data has arrived from stage i
input_activation.requires_grad_(True)
```

The received `input_activation` has `requires_grad=True`. During backward,
stage `i+1` computes `dL/d(activation)` — the gradient with respect to its
input — and sends *that* tensor back to stage `i` as a P2P message:

```python
# Stage i+1: send gradient back
send_buffer, send_work = self._send_grad_async(state.input_activation.grad.detach())

# Stage i: receive gradient, use it as grad_output for backward
grad_output, recv_work = self._recv_grad_async(state.output_activation)
recv_work.wait()
# grad_output is dL/d(output_activation) — the gradient of the loss with respect
# to the tensor that stage i SENT to stage i+1. Calling backward with this value
# propagates gradients through stage i's layers and updates its parameter gradients.
self.runtime._backward_step_impl(grad_output=grad_output)
```

This is the pipeline's "handshake": activations flow forward, gradients flow
backward. The communication involves only two ranks at a time — no collective
operations across all ranks.

---

## Microbatches

If each stage processed one batch at a time, the GPU utilization would be
terrible: stage 0 runs forward, sends its output, then sits idle while stages
1–7 process. Only one stage is busy at a time.

The fix is **microbatches**: split the batch into `M` smaller pieces. Stage 0
processes microbatch 0, sends its output, then immediately starts microbatch 1.
Meanwhile, stage 1 is receiving and processing microbatch 0.

```
Batch of B tokens → M microbatches of B/M tokens each
```

With enough microbatches, every stage has something to work on simultaneously —
up to pipeline depth `P` stages running in parallel. This is what makes PP
efficient.

---

<div class="article-figure">
  <img src="../assets/pp-schedule-diagram.svg" alt="AFAB vs. 1F1B pipeline schedules">
</div>

---

## Schedule 1: AFAB (All-Forward-All-Backward)

The simplest schedule: run all M forward passes first, then all M backward passes.

For stage `s` with `M` microbatches:

```
F(0), F(1), ..., F(M-1), B(M-1), B(M-2), ..., B(0)
```

```python
def _build_afab_schedule(self, num_microbatches: int) -> list[_PipelineAction]:
    actions = [
        _PipelineAction(kind=FORWARD, microbatch_idx=m)
        for m in range(num_microbatches)
    ]
    actions.extend(
        # microbatch_idx: which microbatch's stored activations to use for backward
        # backward_step_idx: sequential index 0, 1, 2, ... used to look up the
        #   corresponding exec_state (gradient buffers) from the ZeRO-3 exec_states list
        _PipelineAction(kind=BACKWARD, microbatch_idx=m, backward_step_idx=i)
        for i, m in enumerate(range(num_microbatches - 1, -1, -1))
    )
    return actions
```

**The bubble problem.** Consider 4 stages and 4 microbatches:

```
Stage 0:  F0  F1  F2  F3  B3  B2  B1  B0  __  __  __
Stage 1:  __  F0  F1  F2  F3  B3  B2  B1  B0  __  __
Stage 2:  __  __  F0  F1  F2  F3  B3  B2  B1  B0  __
Stage 3:  __  __  __  F0  F1  F2  F3  B3  B2  B1  B0
```

Stage 0 starts immediately — it has no *fill* bubble at the front. But after it
finishes B0, it must wait while stages 1–3 drain their remaining backward passes
(the `__` slots at the end of stage 0). Stage 3 has the opposite pattern: it is
idle for the first 3 slots while the pipeline fills, then runs without interruption
until the end. The total idle time — fill idle for later stages, drain idle for
earlier stages — is the pipeline bubble. The exact bubble fraction is derived in
the section below.

**Memory cost of AFAB**: the activations from all M forward passes must be kept
in memory until their corresponding backward pass. A 24-layer model split across
4 stages with M=8 microbatches stores 8 separate sets of intermediate activations
on each stage simultaneously.

---

## Schedule 2: 1F1B (One-Forward-One-Backward)

1F1B interleaves forward and backward passes. After a warmup phase (filling the
pipeline), each stage alternates between processing one forward and one backward:

```python
def _build_1f1b_schedule(self, num_microbatches: int) -> list[_PipelineAction]:
    warmup = min(self.stage_count - self.stage_index - 1, num_microbatches)
    remaining = num_microbatches - warmup  # microbatches that go through the interleaved F/B steady state
    actions: list[_PipelineAction] = []
    # warmup: forward-only until pipeline is filled
    for m in range(warmup):
        actions.append(_PipelineAction(kind=FORWARD, microbatch_idx=m))
    # steady state: interleaved F/B
    for i in range(remaining):
        if warmup + i < num_microbatches:
            actions.append(_PipelineAction(kind=FORWARD, microbatch_idx=warmup + i))
        actions.append(_PipelineAction(kind=BACKWARD, microbatch_idx=i, backward_step_idx=i))
    # drain: remaining backwards
    for i in range(remaining, num_microbatches):
        actions.append(_PipelineAction(kind=BACKWARD, microbatch_idx=i, backward_step_idx=i))
    return actions
```

For stage `s`, warmup = `P - s - 1` forward passes. Stage P-1 (the **last** stage)
has warmup = 0 — it immediately starts backward after its first forward, because
it computes the loss directly. Stage 0 (the **first** stage) has the most warmup
(`P - 1` forwards) because it must fill the entire pipeline before the last stage
can send its first gradient back.

The **bubble ratio** is the same as AFAB — `(P-1)/(M+P-1)` — the total idle time
is identical — but 1F1B distributes the idle time differently and crucially:

**Memory advantage of 1F1B**: at any point in the steady state, each stage holds
at most `P` microbatch activations in memory (one per pipeline stage in flight),
rather than all `M`. For a 4-stage pipeline with M=32 microbatches, AFAB needs
32× microbatch activations; 1F1B needs only 4×. This is the main reason to
prefer 1F1B in production.

---

## The Bubble: How Much Does It Matter?

The standard bubble efficiency formula counts idle slots as a fraction of total
pipeline time. For P stages and M microbatches, a "clock tick" is the time for
one stage to process one microbatch (a single forward or backward pass through
that stage's layers). Total useful ticks = M × 2 (M forwards + M backwards).
Total elapsed time = (M + P - 1) × 2 ticks — M microbatches plus P-1 pipeline
fill/drain steps, each requiring forward and backward. Idle time = (P - 1) × 2
ticks. Simplified, the bubble fraction is:

```
bubble = (P - 1) / (M + P - 1)
```

If you've read the Megatron-LM paper, you may remember the formula as
`(P-1)/M`. Both are correct — they use different denominators. `(P-1)/M` is
the ratio of bubble time to *ideal* (bubble-free) time; `(P-1)/(M+P-1)` is the
bubble's share of *total* elapsed time. For M ≫ P they converge.

At P=4:
- M=4:  3/7  ≈ 43% idle — poor
- M=16: 3/19 ≈ 16% idle — acceptable
- M=32: 3/35 ≈  9% idle — good

In practice, M is chosen to make the bubble acceptable (typically M ≥ 4×P).
But `M` also affects memory: more microbatches means storing more intermediate
activations in AFAB, or increasing the number of in-flight microbatches in 1F1B.
The right `M` is a balance between pipeline efficiency and memory pressure.

---

## Loss and Gradient Averaging

Only the last stage computes a loss. The last stage's loss is broadcast to all
other stages via all-reduce over the PP process group:

```python
def _broadcast_loss(self, total_loss, num_microbatches, device):
    if self.next_global_rank is None:  # True only on the last stage
        # Divide accumulated loss by M to get the average per-microbatch loss
        loss = total_loss / float(num_microbatches)
    else:
        # Non-last stages have no loss; contribute zero to the all-reduce sum
        loss = torch.zeros((), device=device)
    dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.pp_group)
    return loss
```

Each microbatch's loss is divided by `num_microbatches` during backward
(`raw_loss / float(num_microbatches)`), so the total gradient across M
microbatches is equivalent to processing a single batch of size `M × micro_batch_size`.

---

## PP + Other Parallelisms

PP runs first in the plugin ordering — it partitions the model before TP, SP, or
ZeRO transform the layers. The activation tensors crossing stage boundaries need
a defined dtype: MALTOS reads this from the precision plugin (if present) and
casts boundary activations accordingly.

PP + SP has an interaction: with SP active, the activation tensor shape at stage
boundaries is `[B, T/tp_world_size, hidden]` rather than `[B, T, hidden]`. The
PP plugin detects whether SP is active via `sequence_parallel_enabled` and adjusts
the expected buffer shape.

PP + ZeRO: ZeRO-3 has per-microbatch `exec_states` (one per microbatch) to keep
gradient buffers separate across the microbatch accumulation loop. This is why
the ZeRO tutorial mentioned that `exec_states` has one entry per microbatch.

---

## Experiment Placeholder

> **[Placeholder: bubble overhead at different M/P ratios]**
> Benchmark 8-stage PP with M=8 vs. M=32 microbatches using AFAB and 1F1B.
> Expected: 1F1B significantly lower peak memory at M=8 (bounded by stage count,
> not microbatch count). At M=32 the absolute memory differs less but 1F1B
> still wins. Measure: tokens/sec (throughput), `max_memory_allocated()`.

---

## What's Next

The next article covers context parallelism: how to train on very long sequences
by sharding the sequence dimension across GPUs, and why attention — which needs
to see all prior tokens — makes this harder than simple data splitting.

PP scales model depth across nodes. Context parallelism scales sequence length
within and across nodes. The two are orthogonal and can be combined.
