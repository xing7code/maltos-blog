---
layout: post
title: "Putting It All Together: Combining Parallelism Strategies"
description: A map of the seven parallelism strategies in this series — which shard the model, which shard the activations, what each costs — plus how to choose, how they compose into a device mesh, and the very different recipes Qwen3 and DeepSeek-V3 chose to train at scale.
category: Pretraining Concepts · Part 10 of 10
date: 2026-06-18
read_time: 16 min read
---

# Putting It All Together: Combining Parallelism Strategies

The earlier chapters built up the machinery of a training run, and six of them —
Parts 4 through 9 — each introduced a different way to spread that run across GPUs,
motivated by a *different* bottleneck: throughput (DDP), optimizer-state memory
(ZeRO), per-layer weight memory (TP), activation memory at long context (CP), model
depth that won't fit on a node (PP), and parameter count without extra compute
(MoE/EP). In isolation each one is simple. The confusing part — and the part that
decides whether a frontier run is even possible — is that real training combines
several of them at once.

This chapter is the map. It answers three questions the earlier chapters left open:

1. **What does each strategy actually split**, and what does that do to memory?
2. **Which should you reach for**, at a given model size and cluster?
3. **How do they compose** — and what breaks when they do?

Then we look at two real open models — Qwen3 and DeepSeek-V3 — that made
strikingly different choices.

---

## The one question every strategy answers

A single training step has to fit four things on each GPU:

- the **parameters** (`P` numbers),
- their **gradients** (another `P`),
- the **optimizer state** — for Adam, an fp32 master copy of the weights plus two
  moment estimates, together several times the parameters' own footprint and
  usually the single largest consumer,
- the **activations** saved during the forward pass for the backward pass.

Every parallelism strategy is an answer to the same question: *which of these four
do you refuse to replicate across GPUs?* Once you see them that way, the whole
zoo lines up neatly.

| Strategy | What it splits | Across which axis | What each rank still holds whole | Extra communication | Memory it cuts |
|---|---|---|---|---|---|
| **DP / DDP** (Part 4) | the *batch* | data | params, grads, optimizer | gradient all-reduce | nothing — buys throughput |
| **ZeRO-1** (Part 5) | optimizer state | data | params, grads | + gather optimizer step | optimizer state |
| **ZeRO-2** | optimizer + grads | data | params | reduce-scatter grads | + gradients |
| **ZeRO-3 / FSDP** | optimizer + grads + params | data | nothing (re-gathered on demand) | all-gather params, fwd **and** bwd | + parameters |
| **TP** (Part 6) | each weight matrix (rows/cols) | within a layer | activations (partly) | all-reduce twice per block | parameters, and their grads + optimizer |
| **SP** (Part 6) | activations between layers | sequence | params | gather/scatter around TP regions | activation memory |
| **CP** (Part 8) | activations along the sequence | sequence (tokens) | params | ring exchange of K/V | activation memory at long context |
| **PP** (Part 7) | the *layer stack* | depth | — | point-to-point between stages | parameters + activations, per stage |
| **EP** (Part 9) | the *expert pool* | expert | non-expert params | all-to-all token dispatch | expert-parameter memory |

Read the table by its two most important columns. **"What it splits"** tells you
*which bottleneck it attacks*; **"extra communication"** tells you *what you pay*.
There is no free lunch: every row that cuts memory adds a collective.

### Model-sharding vs activation-sharding

Squint at the table and the strategies fall into a few families:

- **Shard nothing, replicate everything** — DP. It only splits the batch, so it
  does nothing for memory; it is the baseline every other strategy layers on top of.
- **Shard the model's *state*** — ZeRO-1/2/3, TP, PP, EP. These cut the memory of
  parameters / gradients / optimizer. ZeRO shards them *along the data axis* while
  keeping each layer's math intact; TP/PP/EP shard the *model itself* so each GPU
  computes only part of every (or only some) layers.
- **Shard the *activations*** — SP and CP (and TP, as a side effect). These attack
  the fourth consumer, which the others ignore and which dominates at long context.

That split is the single most useful mental model: **ZeRO and DP fight over
parameter/optimizer memory along the batch axis; TP/PP/EP cut the model along
structural axes; SP/CP cut the activations along the sequence.** Because they
target different memory, they *stack* — which is exactly why real runs use several
at once.

---

## How each layer actually gets split

The model is not sharded uniformly. Different layer types use different strategies,
and "the model is on 64 GPUs" hides a lot of structure. For a transformer block:

| Component | How it's split under a combined plan |
|---|---|
| **Token embedding** | TP shards the vocabulary dimension; otherwise replicated |
| **Attention (Q/K/V/O)** | TP shards attention *heads*; CP shards the *sequence*; SP shards the activations feeding in and out |
| **Dense FFN** | TP shards the hidden dimension (column-parallel up, row-parallel down) |
| **MoE FFN** | EP shards the *experts* across ranks; each expert is an ordinary MLP that may or may not *also* be TP-sharded |
| **RMSNorm / residual** | parameters replicated; SP shards the activation along the sequence |

Two things to take from this. First, **TP and EP shard different layers** — TP the
attention and dense parts, EP the expert parts — which is why an MoE model reaches
for both. Second, the *sequence* axis (SP, CP) and the *structural* axes (TP, PP,
EP) are orthogonal: a single activation tensor can be sharded by heads *and* by
sequence position at the same time.

---

## Which to reach for

A practical decision order, cheapest and simplest first. Add the next lever only
when the previous one runs out:

| Your situation | Typical configuration |
|---|---|
| Model fits on one GPU, you want speed | DDP (Part 4) |
| Model fits, but optimizer state doesn't | DDP + ZeRO-1/2 (Part 5) |
| Model doesn't fit on one GPU, fits on a node | TP + SP within the node (Part 6), DP/ZeRO across nodes |
| Model doesn't fit on a node | + PP across nodes (Part 7), or ZeRO-3 if the interconnect is fast |
| Training at long context (32K+) | + CP (Part 8) |
| Mixture-of-experts model | + EP (Part 9) |

The ordering encodes two rules of thumb. **Cut optimizer state before you cut the
model** — ZeRO-1 is nearly free (a little extra communication, no change to the
math) and often enough on its own. And **cut the model along the cheapest axis
first**: TP's per-block all-reduce is heavy, so you confine TP to a single node's
fast NVLink and use PP or sharded-DP to cross the slow inter-node network.

A concrete example: a 70B dense model on 64 H100s might run TP=8 (one node), PP=2,
DP=4, with ZeRO-1 on the DP axis — 8 × 2 × 4 = 64 GPUs, each holding 1/16 of the
layers' weights and 1/64 of the optimizer state. These are the same levers exposed
by DeepSpeed (ZeRO stages) and Megatron-LM (TP/PP/CP/EP sizes); the concepts in
this series transfer directly to those frameworks' configuration knobs.

---

## What breaks when you combine them

Stacking strategies is not free composition. The moment you use more than one, the
GPUs stop being a flat list and become a **device mesh** — a grid whose axes are
TP × PP × DP (× CP × EP), and each collective runs along one axis of that grid.
Getting the grid right is most of the difficulty, and it's where a runtime earns
its keep. The recurring hazards:

- **Placement matters as much as size.** TP's all-reduce fires twice per block, so
  its group must sit on the highest-bandwidth link you have — inside a node, on
  NVLink. Put a TP group across nodes and throughput collapses. DP and PP tolerate
  the slow network; TP and EP do not.
- **Gradients can be counted twice.** When a parameter is replicated across one
  axis but sharded across another, the reduction has to average over exactly the
  right group with exactly the right divisor. Get the group wrong and gradients are
  silently scaled — the loss still goes down, just to the wrong place. This is why
  serious tests compare *gradients*, not just loss.
- **EP doesn't get its own GPUs — it reuses an axis.** Expert parallelism has to be
  carved out of a group you already have; frameworks differ on which one (DeepSpeed
  splits it from the data-parallel group, Megatron "folds" the attention layers'
  TP/CP GPUs into the expert region), and expert gradients then reduce over a
  *different* group than the shared parameters.
- **Ordering is a constraint, not a preference.** Some transforms must happen before
  others — you can't shard optimizer state before you know the sharded parameter
  shapes TP produced. A correct runtime enforces this order rather than hoping you
  configured it consistently.

We keep these deliberately shallow here; the mechanics — mesh construction,
gradient-correction factors, transform ordering, and keeping checkpoints correct
across arbitrary combinations — are the subject of the deep dive on
[composable parallelism](../internals/composable-parallelism.html).

---

## Two real recipes

Nothing makes the design space concrete like seeing two strong teams reach
opposite conclusions on the same hardware generation. Both models below are
open-weight mixture-of-experts transformers trained on thousands of GPUs; their
parallelism recipes could hardly be more different.

### Qwen3-235B-A22B — use the whole stack

Qwen3's flagship is a 235B-parameter MoE with 22B active per token (128 experts,
top-8, no shared expert). Alibaba trained it on Megatron-LM with a **hybrid of
every axis in this series**: tensor, pipeline, context, and expert parallelism,
with ZeRO-1 data parallelism on top, at up to ~10,000 GPUs. It is the textbook
five-axis (5-D) parallel run — attention sharded by TP and CP, experts by
EP, depth by PP, optimizer state by ZeRO-1 — each lever pulled exactly where the
earlier chapters said it helps.

### DeepSeek-V3 — deliberately drop TP

DeepSeek-V3 is a much larger MoE — 671B parameters, 37B active (256 routed experts
plus one shared, top-8) — trained on 2,048 H800 GPUs. Its recipe is the surprise:
**16-way pipeline parallelism, 64-way expert parallelism across 8 nodes, and
ZeRO-1 data parallelism — and no tensor parallelism at all.** Instead of leaning on
TP to cut per-layer weight memory, DeepSeek-V3 avoids TP's constant per-block
all-reduce and recovers the memory elsewhere: Multi-head Latent Attention (MLA)
compresses the KV cache, FP8 mixed-precision training shrinks the weight and
activation footprint, and a custom **DualPipe** schedule overlaps the expert
all-to-all with computation so the
cross-node EP traffic is nearly hidden. It even drops the auxiliary load-balancing
loss from [Part 9](moe-and-expert-parallelism.html) in favor of a bias-based,
auxiliary-loss-free balancing scheme.

### Why the difference

Neither recipe is wrong; they optimize the same objective under different
constraints. TP buys memory savings at the price of a bandwidth-hungry all-reduce
on every block — worth it when you have it, painful on a cluster where you'd rather
spend the interconnect on expert routing. DeepSeek-V3 bet that MLA + FP8 + a
bespoke pipeline could reclaim TP's memory savings *without* TP's communication, so
it spent its network budget on EP instead. Qwen3 took the well-trodden path and
tuned every knob. The lesson of this series is exactly that: these are **levers,
not a recipe** — the right combination depends on your model's shape and your
cluster's wiring, and the frameworks you'll use (DeepSpeed, Megatron-LM) expose all
of them precisely so you can choose.

---

## The end of the series

Across ten chapters we built up from a single-GPU training loop to the full
vocabulary of large-scale pretraining: the data pipeline, distributed primitives,
and the seven parallelism strategies that frontier runs combine. You now have the
mental model to read a model's technical report and understand *why* it was trained
the way it was — and the map to choose, and compose, these strategies for a run of
your own.

For how a runtime makes these strategies compose *correctly* — the plugin system,
the optimizer-factory pattern, gradient-correction factors, and checkpoints that
stay valid under any combination — continue to the [deep-dive
series](../internals/).
