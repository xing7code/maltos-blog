---
layout: post
title: "Putting It All Together: Combining Parallelism Strategies"
description: A clear map of the seven parallelism strategies in this series, worked out on a modern LLM layer (GQA, SwiGLU, RMSNorm, FlashAttention) — which ones cut parameter memory, which cut activation memory, how much each sends over the network per layer, and where to place each on the cluster — with a how-to-choose guide and two real examples, Qwen3-235B and DeepSeek-V3.
category: Pretraining Concepts · Part 10 of 10
date: 2026-06-18
read_time: 20 min read
---

# Putting It All Together: Combining Parallelism Strategies

The last six chapters (Parts 4–9) each showed one way to split training across GPUs,
and each one solved a different problem. This chapter puts them side by side with
numbers: how much memory each one saves, how much network traffic it adds, where to
place it on the cluster, and how two real models put it all together.

Everything below is about **one transformer layer** of a modern LLM. The symbols:

- `B` — batch;
- `S` — sequence length;
- `L` — number of layers;
- `D` — model dim;
- `H` — query heads;
- `Hkv` — key/value heads;
- `d = D/H` — per-head dim;
- weights and activations are `bf16` (2 bytes throughout).

We also fix a modern architecture (like Llama, Qwen, or DeepSeek), because the exact
numbers depend on it:

- **GQA attention** — `H` query heads but fewer KV heads, `Hkv` (`Hkv ≤ H`). The Q and
  output weights are `D×D`; the K and V weights are smaller, `D×(Hkv·d)` — shrunk by
  the ratio `Hkv/H`.
- **SwiGLU MLP** — three matrices: `up` and `gate` (`D → 2.7D`) and `down`
  (`2.7D → D`). The width `2.7D` (that's `8/3·D`) is picked so these three matrices
  have the same `8D²` parameters as the old two-matrix `4D` FFN.
- **RMSNorm** with pre-norm residuals, and **FlashAttention** — so the `S×S` attention
  scores are never stored; they are recomputed in the backward pass.

Two facts to keep in mind — we work them out below. A dense layer has about
**`10D²`–`12D²` parameters** (`8D²` in the MLP, `2D²`–`4D²` in attention, less with
GQA). And its stored activations grow **linearly with `B·S·D`** — with FlashAttention
there is no `S²` memory term at all.

---

## Cut parameters, or cut activations?

Two different things fill up GPU memory:

- **Parameter memory** — the weights, plus their gradients and optimizer state. It
  grows with `D²` and the number of layers, but **not** with batch or sequence length.
- **Activation memory** — the tensors the forward pass saves for the backward pass. It
  grows with `B·S·D`, but **not** with the size of the weights.

Sorting the strategies by what they shrink gives three groups:

- **Cut parameters *and* activations — TP, PP, EP.** They split the weights or the
  layers, and the activations of that split-up work shrink along with them.
- **Cut parameters only — ZeRO.** It shards the optimizer state, gradients, or weights
  across data-parallel ranks, but each rank still runs a full forward pass, so its
  activations are untouched.
- **Cut activations only — SP, CP.** They leave every weight whole and shrink only the
  activations.

The parameters-vs-activations split is still the right first cut: real runs need both,
because splitting the weights does nothing when long sequences fill memory with
activations, and splitting the activations does nothing when the weights don't fit.
Let's take each side in turn.

---

## Cutting parameter memory

First, where the parameters live. Per layer:

- **Attention** = `2(1 + Hkv/H)·D²`. Q and output are `D²` each; K and V are
  `(Hkv/H)·D²` each — smaller because of GQA. (Plain MHA has `Hkv = H`, giving the
  familiar `4D²`.)
- **MLP (SwiGLU)** = `3 × 2.7D² = 8.1D²`.

So one dense layer holds `P = 2(1 + Hkv/H)·D² + 8.1D²`, about `10D²` with GQA. Each of
the four parameter-cutting strategies splits a **different part** of this. The table
shows the part each one touches, not the whole layer:

| Strategy (degree) | Weights it touches | Per-layer params **before → after** |
|---|---|---|
| **TP** (t) | every attention + MLP linear | `P → P / t` |
| **EP** (e) | the expert MLPs only | `3m·N·D² → 3m·N·D² / e` |
| **ZeRO-3 / FSDP** (z) | every weight (resident copy) | `P → P / z` (re-gathered to compute) |
| **PP** (p) | whole layers | `L·P → (L/p)·P` per rank |
| **ZeRO-1 / 2** | none — weights stay whole | `P → P` (shards optimizer state / grads) |

The **EP** row needs the MoE picture. An MoE layer replaces the one MLP with `N`
experts. Modern MoEs are *fine-grained*: each expert is narrow, with inner width `m·D`
where `m` is often about `1` or less (with many experts, no single one has to be wide).
So one expert is `3m·D²`, and all the experts in a layer add up to `3m·N·D²` — that is
the part EP splits across ranks. Attention and the router stay the same.

Reading it by the weights each strategy touches makes the differences clear:

- **TP and ZeRO-3** both split the weights they touch by their degree. The difference:
  TP keeps its shard and computes on it directly, while ZeRO-3 only stores its shard and
  re-gathers the full weight each time the layer runs — it pays extra communication to
  hold less at rest.
- **EP** touches only the experts. In a layer with many experts, `3m·N·D²` is far bigger
  than the attention, so splitting just the experts takes care of almost all the memory.
  That is why EP is the first-class citizen for MoE.
- **PP** shrinks weights too, but a whole layer at a time: each rank holds only `L/p`
  layers, so it stores fewer *whole* layers — it just doesn't make any single layer
  smaller.
- **ZeRO-1/2** are the only ones that shrink no weight at all: they keep every weight
  whole and instead split the optimizer state and gradients.

---

## Cutting activation memory

Now the other side. What does one layer save for the backward pass? With FlashAttention
the `S×S` scores are gone, so everything left is either a projection input or an MLP
intermediate — and all of them are `∝ B·S·D`:

| Stored tensor | Size (in `B·S·D`) |
|---|---|
| Attention — layer input, Q, K, V, output | `1 + 1 + (Hkv/H) + (Hkv/H) + 1 = 3 + 2·(Hkv/H)` |
| MLP — norm output, `up`, `gate`, `SiLU(gate)⊙up` | `1 + 2.7 + 2.7 + 2.7 = 9.1` |

Each attention term is one stored tensor, in units of `B·S·D`. The **layer input**,
the **Q** projection, and the **attention output** (what feeds the output projection)
are full width — `1` each, giving the `3`. **K** and **V** are smaller, because GQA
uses only `Hkv` heads — `Hkv/H` each, giving the `2·(Hkv/H)`. These are exactly the
tensors FlashAttention needs for its backward pass; the `S×S` scores are recomputed,
not stored. (The MLP's `2.7`s are the three SwiGLU intermediates, each `2.7D` wide.)

That adds up to about **`12·B·S·D` elements (`24·B·S·D` bytes) per layer**. Most of it is
the three SwiGLU intermediates, and there is **no `S²` term**. So on a modern model,
activation memory grows linearly with sequence length, and it lives mostly in the wide
MLP — not in attention.

Here is how much of that activation each rank keeps, and along which dimension the
split happens:

| Strategy (degree) | Activation left per rank | Dimension it splits |
|---|---|---|
| **TP alone** (t) | the inner attention + MLP activations `/ t`; norm, residual, and layer I/O stay whole | head / hidden dim |
| **TP + SP** (t) | **all** activations `/ t` | + sequence dim (the outer / connecting activations) |
| **CP** (c) | **all** activations `/ c` | sequence dim |

TP and SP cut along *different* dimensions, which is why they pair up so well. **TP**
splits the work *inside* attention and the MLP (along the head and hidden dims), so the
inner activations shrink — but the norm, residual, and layer input/output between the
blocks stay copied on every rank. **SP** splits exactly those outer activations, along
the *sequence*. Together, TP+SP divide **all** activations by `t`. **CP** also splits the
sequence, and it's the tool for long context: FlashAttention already removed the `S²`
*memory*, but activations still grow with `S` and the attention *compute* is still
`O(S²)` — CP splits both across ranks.

---

## What each strategy costs to communicate

Saving memory always costs network traffic, and that cost decides where each strategy
goes on the cluster. We can compute the exact byte count from two facts. 

* First, the two
tensor sizes: a layer's **activation** is `act = 2·B·S·D` bytes, and its **weights** are
`2P ≈ 20D²` bytes. 
* Second, the cost of a ring collective — this is not a guess, it's how
the algorithm works: an **all-reduce moves `2×`** its tensor per rank (it's a
reduce-scatter plus an all-gather), while an **all-gather, reduce-scatter, or all-to-all
moves `1×`**. Put the right tensor into the right collective and you get the traffic per
rank:

| Strategy | Collectives (fwd / bwd) | Tensor moved | Bytes / rank | Fires |
|---|---|---|---|---|
| **DP / DDP** | — / 1 all-reduce | gradients (`20D²`) | `2 × 20D² = ` **`40D²`** | once / step, overlaps backward |
| **ZeRO-1** | — / reduce-scatter + all-gather | grads then params | `20D² + 20D² = ` **`40D²`** | once / step |
| **ZeRO-3 / FSDP** | all-gather / all-gather + reduce-scatter | weights (`20D²`) | `3 × 20D² = ` **`60D²`** | **every layer** |
| **TP** | 2 all-reduce / 2 all-reduce | activations (`act`) | `4 × 2·act = ` **`16·B·S·D`** | **every layer** |
| **SP** (with TP) | 2 all-gather + 2 reduce-scatter / same | activations | `8 × act = ` **`16·B·S·D`** (same as TP) | **every layer** |
| **CP** (c) | ring K,V, `c−1` hops each pass | K,V, GQA-shrunk | **`≈ 8·(Hkv/H)·B·S·D`** | **every layer**, overlappable |
| **PP** (p) | 1 send / 1 recv (point-to-point) | activations | **`≈ 4·B·S·D`** | once / step (at its boundary) |
| **EP** (e) | 2 all-to-all / 2 all-to-all | tokens, top-`k` | **`≈ 8k·B·S·D`**, independent of `N` | **every MoE layer** |

(The `20D²` uses the GQA layer size `P ≈ 10D²`; `EP`'s `k` is the routing top-`k`.)

The **CP** number decomposes as `2 × 2 × 2 · (Hkv/H)·B·S·D`: **K and V** (`×2`), **`bf16`
bytes** (`×2`), and **forward + backward** (`×2`), times the size of one K tensor,
`(Hkv/H)·B·S·D`. Note it does *not* depend on the CP degree `c` — each rank passes the
whole sequence's K/V around the ring no matter how finely it is split — and GQA's
`Hkv/H` is what makes it the cheapest per-layer collective (at `Hkv/H = 1/8`, only
`~1·B·S·D` per layer, versus TP's `16·B·S·D`).

Two things matter together — how often it fires, and how big it is:

- **TP, ZeRO-3, and EP fire on every layer, on the critical path.** TP does four
  activation all-reduces; ZeRO-3 moves a full layer's weights in both passes; EP does
  four token all-to-alls per MoE layer. They fire often and are hard to overlap with
  computation.
- **DP and ZeRO-1 fire once per step**, and their reduction overlaps the backward pass,
  so the cost is mostly hidden.
- **PP communicates only at stage boundaries** — a few point-to-point sends per
  microbatch, not per layer.
- **CP is the cheapest of the per-layer strategies under GQA**: it moves only K and V,
  which GQA has already shrunk by `Hkv/H`, and it overlaps with attention compute.

---

## Which collective goes where

Traffic isn't just bytes — it's bytes over a *link*, and links differ by 10× or more
(NVLink inside a node ≈ 900 GB/s, InfiniBand between nodes ≈ 50–100 GB/s per direction).
The table above tells you which strategy has to sit on the fast link:

| Priority | Strategy | Why | Place it |
|---|---|---|---|
| **1 (fastest link)** | **TP + SP** | 4 activation all-reduces per layer, on the critical path, hard to overlap | **inside a node** (NVLink) |
| **1** | **EP** | 2–4 token all-to-alls per MoE layer; needs low latency *and* high bandwidth | inside a node, or overlap with a custom schedule to cross nodes |
| **2** | **ZeRO-3 / FSDP** | full-weight all-gather every layer, but *can* overlap with compute | prefers a fast link; can cross nodes if bandwidth is high |
| **3** | **CP** | small (K/V only, shrunk by GQA) and overlaps attention compute | can cross nodes; each hop adds a ring step |
| **4 (slow link OK)** | **PP** | point-to-point, only at stage boundaries | **across nodes** — designed for it |
| **4** | **DP / ZeRO-1** | one reduction per step, overlaps backward | **across nodes** |

This is the most important rule when you lay out a run, and it's why the standard recipe
is **"TP inside the node, PP and DP across nodes."** You spend the fast NVLink on the
strategy that uses it every layer (TP), and put the rare or point-to-point traffic
(DP, PP) on the slow between-node link, where it has time to hide. EP is the exception:
its all-to-all is as heavy as TP, so frameworks either keep the experts inside a node,
or — like DeepSeek-V3 below — use a custom schedule to overlap the cross-node expert
traffic with computation.

(Making all of this *correct* once combined — reducing each gradient over the right
group, applying the shard steps in the right order, keeping checkpoints valid — is a
separate problem, covered in the [composable parallelism deep
dive](../internals/composable-parallelism.html).)

---

## Choosing a combination

Here's a simple order to decide, cheapest first. Add the next tool only when the current
one runs out:

| Your situation | Reach for | Main new cost |
|---|---|---|
| Model fits on one GPU, want throughput | **DDP** (Part 4) | grad all-reduce, once / step |
| Fits, but optimizer state doesn't | **+ ZeRO-1/2** (Part 5) | ≈ DDP, near-free |
| A layer's weights don't fit on one GPU | **+ TP + SP** within a node (Part 6) | activation all-reduce every layer → keep inside a node |
| The whole stack doesn't fit on a node | **+ PP** across nodes (Part 7), or **ZeRO-3** if the link is fast (Part 5) | PP: pipeline bubble; ZeRO-3: weight all-gather every layer |
| Context length (32K+) strains activations | **+ CP** (Part 8) | ring K/V exchange, overlappable |
| Mixture-of-experts model | **+ EP** (Part 9) | token all-to-all every MoE layer → keep inside a node |

The order follows two rules. **Shrink the optimizer state before you split the model** —
ZeRO-1 is almost free. And **split the model along whichever axis is cheapest to place** —
TP only where you have NVLink, PP and DP to cross the slow network.

---

## Two real recipes

Two open MoE models, the same GPU generation, opposite choices. First, why the memory
forces the issue at all: with mixed-precision Adam, every parameter costs about
**16 bytes** — `2` for the `bf16` weight, `2` for its gradient, and **`12` for the
optimizer state** (an fp32 master copy plus two moments). The optimizer state alone is
the dominant term, and it is why sharding is not optional at this scale.

### DeepSeek-V3 — deliberately drop TP

**Model.**

| | |
|---|---|
| Total / active params | 671B / 37B (~5.5% run per token) |
| Layers | 61 (MoE in all but the first 3) |
| Model dim `D` | 7168 |
| Attention | **MLA** — 128 heads, K/V compressed to a 512-dim latent |
| Experts | 256 routed + 1 shared, top-8, expert inner dim 2048 (fine-grained) |
| Context | 128K tokens (extended via YaRN) |
| Precision | FP8 training |

**Memory.** The optimizer state dominates. DeepSeek-V3 keeps an **FP32 master weight**
plus the two Adam moments in **BF16** — `4 + 2 + 2 = 8` bytes per parameter — so for 671B
parameters the optimizer alone is **~5.4 TB**. Add the `bf16` weights and gradients (~1.3
TB each) and the run needs **~8 TB** for weights + grads + optimizer. FP8 (below) is a
*compute* precision — it does **not** shrink these high-precision copies — so that ~8 TB
still has to be split across GPUs; no single 80 GB GPU comes close.

*What FP8 does, briefly:* only the big linear-layer matmuls run in 8-bit — their inputs are
cast to FP8 for roughly `2×` the tensor-core throughput and half the activation and
communication bytes, but each product still **accumulates in FP32**. The sensitive parts
(master weights, optimizer moments, norms, attention, embeddings) stay in BF16/FP32. So FP8
buys compute speed and activation memory, which is why it leaves the numbers above unchanged.

**Parallelism.** The setup is the surprise: **16-way pipeline parallelism, 64-way expert
parallelism across 8 nodes, ZeRO-1 data parallelism — and no tensor parallelism at all**,
on **2,048 H800 GPUs** (256 nodes × 8). Here is how the mesh shards that ~8 TB:

- **PP = 16** splits the 61 layers into 16 stages (~4 layers each). Since `2048 = 16 × 128`,
  each stage is 128 GPUs — that is, **DP = 128** data-parallel replicas of the pipeline.
- **EP = 64 is carved out of that 128**, not a separate axis. The 256 experts split 64 ways
  — **4 experts per GPU**, each EP group spanning 8 nodes — so the full expert set sits in
  `128 / 64 = 2` copies (the *expert-data-parallel* replicas). The shared weights (MLA
  attention, embeddings) are instead replicated across all 128 and reduced data-parallel.
- **No TP.** MLA shrinks the KV, FP8 halves the weights, and fine-grained experts keep each
  expert small, so a layer's non-expert weights fit on one GPU without splitting them —
  which lets DeepSeek-V3 skip TP's heavy per-layer all-reduce entirely. That choice was all
  the more attractive on the **H800** — an export-limited H100 whose NVLink is cut to
  roughly half (~400 vs. ~900 GB/s), the very intra-node link TP would hammer.
- **ZeRO-1** shards the ~5.4 TB optimizer state over the data-parallel ranks — 128-way for
  the shared weights, and over the 2 expert replicas for the experts.
- **DualPipe**, a two-direction pipeline schedule, overlaps the cross-node expert all-to-all
  with computation — the priority-1 traffic TP would otherwise be fighting for.

Concretely, one GPU (holding ~4 layers, and 4 of the 256 experts per layer) ends up with:

| Per GPU (~4 layers) | Weights (`bf16`) | Optimizer (ZeRO-1) |
|---|---|---|
| **Shared** — MLA + shared expert + router; replicated, optimizer sharded **128-way** | ~1.9 GB (932M params) | `932M / 128` → **~58 MB** |
| **Experts** — 16 of the 256, EP-sharded; optimizer sharded only **2-way** (EDP) | ~1.4 GB (704M params) | `704M / 2` → **~2.8 GB** |
| **Total** | **~3.3 GB** | **~2.9 GB** |

*How each figure is computed* — with `D = 7168`, `L/PP ≈ 4` layers per GPU, `EP = 64`,
`DP = 128`, `EDP = 2`, and per-parameter bytes `w = 2` (the `bf16` weight) and `o = 8`
(the optimizer: fp32 master + two `bf16` moments):

- one expert `= 3·D·2048 = 44M` params; a GPU holds `(256/EP)·(L/PP) = 4×4 = 16` of them
  → `704M` expert params.
- one layer's shared params (MLA `≈ 187M` + shared expert `44M` + router `D·256 ≈ 2M`)
  `≈ 233M`; a GPU holds all `4` layers → `932M`, replicated across the 128.
- **weights** `= params · w`. **optimizer** `= params · o / (ZeRO-1 group)` — and that
  group is `DP = 128` for shared params (`932M/128 · 8 = 58 MB`) but only `EDP = 2` for
  experts (`704M/2 · 8 = 2.8 GB`).

The last column is the punchline: the experts are split 64 ways by EP, but their optimizer
is split only *two* ways (over the EDP replicas) — so it, not the 128-way-sharded shared
optimizer, is what dominates each GPU's memory. Get that group wrong and the ~5.4 TB
optimizer would not fit. All told each GPU carries ~6 GB of weights + optimizer, plus
gradients and activations — comfortably inside 80 GB. DeepSeek-V3 even replaced the
auxiliary load-balancing loss from [Part 9](moe-and-expert-parallelism.html) with a
bias-based, auxiliary-loss-free scheme.

**Takeaway.** We can't know DeepSeek's exact motivations — MLA, FP8, and fine-grained
experts each have their own reasons (a small KV cache, faster compute, more expert
specialization). But together they leave a layer's weights small enough that TP isn't
needed, and the team skipped it. That's the real lesson: architecture and parallelism
aren't separate decisions — a memory-lean model can drop the most placement-constrained
collective and spend its bandwidth on EP instead.

### Qwen3-235B-A22B — use the full stack

**Model.**

| | |
|---|---|
| Total / active params | 235B / 22B |
| Layers | 94 (all MoE) |
| Model dim `D` | 4096 |
| Attention | GQA — 64 query heads, 4 KV heads (`Hkv/H = 1/16`), head dim 128 |
| Experts | 128, top-8, no shared expert, expert inner dim 1536 |
| Context | 128K tokens |

**Where the parameters live (per layer, ≈2.5B):**

| Component | Per-item | Params |
|---|---|---|
| 128 experts | `3·D·1536 = 18.9M` each | **2.42B** |
| GQA attention | Q, O (`D×8192`) `33.6M` each; K, V (`D×512`) `2.1M` each | ~71M |
| router | `D·128` | 0.5M |

That is `~2.49B` per layer, and `× 94 ≈ 234B` (plus embeddings ≈ 235B). The experts are
`~97%` of it — the same lopsided split as DeepSeek, which is again why EP is the lever.

**Memory.** 235B parameters → **470 GB of `bf16` weights** and a **~2.8 TB optimizer state**
(standard fp32-moment Adam, `12 B/param`) — again far past one GPU, again optimizer-dominated.

**Parallelism.** Alibaba trained it on Megatron-LM across a **five-axis mesh**
(TP × PP × CP × EP × DP), with ZeRO-1 on the DP axis, scaling to on the order of
**10,000 GPUs**. The report does **not publish the exact degrees**, so — unlike DeepSeek —
there is no public per-GPU breakdown to work through. But each axis plays the role the
sections above predict: TP + SP splits each layer *inside a node* (priority 1); PP splits
the 94 layers *across nodes*, where boundary traffic is cheap (priority 4); EP spreads the
128 experts (priority 1, on fast links); CP handles the 128K context (priority 3); and
ZeRO-1 shards the 2.8 TB optimizer over the DP ranks. And the reasoning transfers even
without the numbers: whatever the EP degree, each GPU holds `128 / EP` experts, and — exactly
as in DeepSeek's table — their optimizer is sharded only over the small expert-replica group,
so it dominates per-GPU memory. One more effect of the aggressive GQA (`Hkv/H = 1/16`): the
K and V are `16×` smaller than the queries, so both the attention activations and CP's ring
traffic are tiny.

**Takeaway.** With a balanced cluster and a standard GQA + MoE model, using every tool is
the safe, well-understood default — and it scales to five axes cleanly.

### Why they differ

Neither choice is wrong; they solve the same problem in different ways. TP saves weight
memory, but it costs the heaviest collective and it has to stay inside a node. Qwen3 paid
that cost and kept TP on NVLink; DeepSeek-V3's model was lean enough (MLA, FP8, fine-grained
experts) to skip TP entirely.
**These are tools, not a fixed recipe.** The right mix is whatever makes your model fit
your cluster, and every framework you'll use (DeepSpeed, Megatron-LM) gives you all of them
so you can choose.

---

## The end of the series

Over ten chapters we went from a single-GPU training loop to the full toolbox of
large-scale pretraining. You can now open a model's technical report, read its parallelism
setup, and work out *why* — which memory it was saving, which collective it was paying for,
and why each axis sits where it does on the cluster.

To see how a runtime makes these strategies work together *correctly* — the plugin system,
the optimizer-factory pattern, gradient-correction factors, and checkpoints that stay valid
in any combination — continue to the [deep-dive series](../internals/).
