---
layout: post
title: "Pipeline Parallelism"
description: When a model is too large to fit on a node — or when tensor parallelism has hit its scaling limit — pipeline parallelism splits the model vertically across stages and keeps all GPUs busy with a schedule of microbatches. This article covers how the split works, the two common schedules, and where the bubble comes from.
category: Pretraining Concepts · Part 7 of 10
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

A transformer has three kinds of layers, and they split differently:

- **Head layers** — the token embedding and any input-side setup (e.g. RoPE
  tables). These live only on the first stage, because it is the only stage that
  sees raw token IDs.
- **Transformer blocks** — the bulk of the model. These are partitioned across
  stages: the first few blocks on stage 0, the next few on stage 1, and so on.
- **Tail layers** — the final layer norm and the LM head. These live only on the
  last stage, which produces the logits and computes the loss.

The transformer blocks are usually partitioned evenly. For a 24-layer model
across 4 stages, each stage gets 6 blocks. When the layer count doesn't divide
evenly, the simplest policy is to give the first few stages one extra block each
— for a 25-layer model across 4 stages, stage 0 gets 7 and stages 1–3 get 6.

Even splitting is only an approximation of balanced work, though. Stage 0 also
carries the embedding, and the last stage carries the LM head — which for large
vocabularies can cost as much as several transformer blocks. So production
systems often balance stages by *compute and memory* rather than by raw block
count, shifting a block or two off the heavier end stages. The rest of this
article assumes an even split for simplicity.

Whatever the split, each stage only allocates and runs the layers assigned to it.
The layers belonging to other stages are not loaded on this device — a stage holds
just its own slice of the parameters, which is exactly what buys the memory saving.

---

## What Crosses Stage Boundaries

Only stage 0 receives raw token IDs as input. Every stage after that receives
a hidden state tensor — the output of the previous stage's last transformer
block — and never sees the original vocabulary tokens.

Each stage's output is likewise a hidden state tensor: the activation after the
last transformer block on that stage. Stage `i` sends its output to stage `i+1`
using a point-to-point (P2P) send — a message between exactly two ranks, not a
collective across the whole pipeline. The send is typically **asynchronous**:
stage `i` enqueues the transfer and can immediately start its next piece of work
(for example, the forward pass of the next microbatch) instead of blocking until
the data lands on stage `i+1`. Stage `i+1` only waits when it actually needs the
tensor to compute.

```python
# Stage i: send activations to the next stage (non-blocking)
send_work = send_forward(boundary_activation)

# Stage i+1: receive activations from the previous stage
input_activation, recv_work = recv_forward()
recv_work.wait()                      # block until the data has arrived
input_activation.requires_grad_(True) # this stage's input into the graph
```

**The backward direction is where the asymmetry between stages shows up.** Only
the last stage has a loss to differentiate: it holds the LM head, computes the
logits, and evaluates the loss for each microbatch. So the last stage is the one
that calls `loss.backward()` — that is what kicks off the entire backward pass.
It has no downstream neighbor to receive a gradient from; it starts from the loss
itself.

Every other stage has no loss of its own. A stage in the middle only learns how
to run its backward pass once its *downstream* neighbor hands back a gradient. The
received input activation carries `requires_grad=True`, so during backward stage
`i+1` computes `dL/d(activation)` — the gradient with respect to its input — and
sends *that* tensor back to stage `i`:

```python
# Stage i+1: send the input-activation gradient back upstream
send_work = send_backward(input_activation.grad)

# Stage i: receive that gradient and use it as grad_output
grad_output, recv_work = recv_backward()
recv_work.wait()
# grad_output is dL/d(output_activation): the gradient of the loss w.r.t. the
# tensor stage i SENT downstream. Running backward from it propagates gradients
# through stage i's layers and accumulates its parameter gradients.
output_activation.backward(grad_output)
```

So the loss lives on exactly one stage, and every earlier stage is driven purely
by the activation gradient arriving from the stage in front of it. This is the
pipeline's "handshake": activations flow forward, gradients flow backward, and
each hop touches only two neighboring ranks.

---

## Microbatches

If each stage processed one batch at a time, the GPU utilization would be
terrible: stage 0 runs forward, sends its output, then sits idle while stages
1–7 process. Only one stage is busy at a time.

The fix is **microbatches**: split the batch into `M` smaller pieces. Stage 0
processes microbatch 0, sends its output, then immediately starts microbatch 1.
Meanwhile, stage 1 is receiving and processing microbatch 0.

With enough microbatches, every stage has something to work on simultaneously —
up to pipeline depth `P` stages running in parallel. This is what makes PP
efficient.

One thing to get right when splitting a batch this way: the **gradient scale**.
If the loss is a per-token (or per-sample) average — the usual default — then each
microbatch's backward already produces the average gradient over *its own* tokens.
Running backward on all `M` microbatches accumulates those gradients into the
parameters, giving the *sum* of `M` per-microbatch averages — which is `M` times
too large. To recover the true batch average, each microbatch's loss is scaled by
`1/M` before backward. This is exactly the gradient-accumulation normalization
from earlier in the series; microbatches are just one more place it shows up.

---

## Scheduling

With `P` stages and `M` microbatches, there are `P × M × 2` pieces of work to run:
every microbatch does a forward and a backward pass on every stage. Two
dependency rules constrain the order:

- **Forward:** stage `N`'s forward for a microbatch can't start until stage `N-1`
  has produced that microbatch's activation.
- **Backward:** stage `N`'s backward can't start until stage `N+1` has sent back
  the gradient for that microbatch.

A **schedule** is any ordering of those `P × M × 2` ops that respects both rules.
Within that freedom there's a lot of room: which microbatch a stage works on next,
and whether it does forward or backward, is a choice. A *good* schedule makes that
choice so as to keep every stage as busy as possible (minimal **bubble**) while
holding as few activations in memory as possible.

Two schedules are enough to see the whole trade-off — **AFAB** and **1F1B**. The
bubble is essentially fixed by the ratio of `P` to `M` (derived below), not by
which of these two you pick — so both finish in the same time and carry the same
bubble. What differs sharply is peak memory. The diagram shows both:

<div class="article-figure">
  <img src="../assets/pp-schedule-diagram.svg" alt="AFAB vs. 1F1B pipeline schedules">
</div>

One simplification in the diagram: forward and backward cells are drawn the same
width. In reality a **backward pass costs roughly twice a forward pass**, because
it computes two gradients — the gradient w.r.t. the layer's *input* (to pass
upstream) and the gradient w.r.t. its *weights* (to update parameters) — where the
forward computes only the output. The equal widths are just for legibility. This
2:1 split is also what some advanced schedules exploit: because the input-gradient
and weight-gradient halves are independent, they can be reordered separately to
fill idle slots, which is the core idea behind zero-bubble scheduling below.

**AFAB (All-Forward-All-Backward)** — introduced by GPipe
([Huang et al., 2018](https://arxiv.org/abs/1811.06965)) — is the simplest: each
stage runs all `M` forward passes, then all `M` backward passes.

```text
per stage:  F(0) F(1) ... F(M-1)   B(M-1) B(M-2) ... B(0)
```

The backward order is reversed because the last activation produced is the first
one whose gradient comes back. AFAB is easy to reason about, but it has a memory
problem: every forward pass stores its activations, and none of them can be freed
until the matching backward runs. So each stage holds **all `M`** microbatches'
activations at once — the green bracket in the diagram.

**1F1B (One-Forward-One-Backward)** — introduced by PipeDream
([Harlap et al., 2018](https://arxiv.org/abs/1806.03377)) and used in the
synchronous, flushed form described here by Megatron-LM
([Narayanan et al., 2021](https://arxiv.org/abs/2104.04473)) — interleaves the
two. Each stage first does a short warmup of forward-only passes to fill the
pipeline, then settles into a steady state of one forward followed by one
backward, and finally drains the remaining backwards.

```text
per stage s:  warmup:        (P - 1 - s) forward-only passes
              steady state:  F, B, F, B, ...  (one-forward-one-backward)
              cooldown:      the remaining backward passes
```

The warmup length depends on the stage. The last stage has no warmup — it starts
backward right after its first forward, because it owns the loss. The first stage
warms up the longest (`P-1` forwards), since it must fill the whole pipeline
before the last stage can send a gradient back.

The payoff is memory. As soon as a microbatch's backward runs, its activations
are freed, so the steady state keeps **at most `P`** microbatches in flight rather
than all `M`. For `P=4`, `M=32`, AFAB stores 32× activations per stage while 1F1B
stores only 4×. That is why 1F1B is the default in practice.

Crucially, **1F1B does not run faster** — the diagram shows both finishing in the
same 18 ticks, with the same total idle time. It rearranges *when* the idle slots
fall, not how many there are. The idle time itself is the bubble, quantified next.

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

## Beyond AFAB and 1F1B

AFAB and 1F1B are the two schedules worth understanding in full, but they are the
floor, not the ceiling. Production frameworks push the bubble lower with more
elaborate schedules — worth knowing by name so you recognize them:

- **Interleaved 1F1B** ([Narayanan et al., 2021](https://arxiv.org/abs/2104.04473))
  — give each device *several* non-contiguous stages (e.g. layers 0–3 and 16–19)
  instead of one contiguous block. This shrinks the bubble roughly by the number
  of chunks per device, at the cost of more cross-stage communication. It is the
  standard schedule in Megatron-LM. Within this interleaved setting there's a
  further choice of how to order the microbatches: the default pushes each
  microbatch as deep as it can go (depth-first), while **breadth-first / looping**
  schedules ([Lamy-Poirier, 2023](https://arxiv.org/abs/2211.05953)) instead
  advance many microbatches one chunk at a time, filling the pipeline more fully
  and overlapping cross-node communication better — at the cost of higher
  activation memory.
- **Zero-bubble / ZB-H1** ([Qi et al., 2023](https://arxiv.org/abs/2401.10241))
  — split the backward pass into its two independent halves (gradient w.r.t. input
  vs. gradient w.r.t. weights) and reorder them to fill idle slots, driving the
  steady-state bubble toward zero.
- **DualPipe** ([DeepSeek-V3, 2024](https://arxiv.org/abs/2412.19437)) — a
  *bidirectional* schedule that feeds microbatches in from both ends of the
  pipeline at once, so forward and backward streams overlap and nearly cancel the
  bubble. The catch is that it keeps two copies of the model's stages in play,
  roughly doubling the parameter memory — a trade it makes to hide communication
  on large clusters.

All of them keep the same core idea — activations forward, gradients backward,
neighbor-to-neighbor P2P — and differ only in the *order* of operations. AFAB and
1F1B are enough to understand the trade-off; the rest are refinements of it.

---

## PP + Other Parallelisms

PP is largely orthogonal to the other parallelism strategies: it splits the model
*across* stages, while TP, SP, and ZeRO change how each stage's layers are stored
and computed *within* a stage. In a full 3D setup, PP decides which layers live on
which stage first, and the other strategies then shard those layers.

The one interaction worth flagging is **PP + sequence parallelism**. SP shards
activations along the sequence dimension, so the hidden state crossing a stage
boundary is `[B, T/tp, hidden]` rather than the full `[B, T, hidden]`. The send
and receive buffers on either side of the boundary have to agree on this sharded
shape — a stage expecting `[B, T, hidden]` while its neighbor sends
`[B, T/tp, hidden]` is a size mismatch that will fail at communication time. When
combining PP with SP, make sure the boundary buffer shapes account for the
sequence sharding.

---

## What's Next

The next article covers context parallelism: how to train on very long sequences
by sharding the sequence dimension across GPUs, and why attention — which needs
to see all prior tokens — makes this harder than simple data splitting.

PP scales model depth across nodes. Context parallelism scales sequence length
within and across nodes. The two are orthogonal and can be combined.
