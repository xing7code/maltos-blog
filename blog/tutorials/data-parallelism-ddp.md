---
layout: post
title: "Data Parallelism and Gradient Reduction"
description: DDP is the first parallelism strategy every pretraining setup needs. This article covers synchronous gradient reduction, per-parameter async overlap, and the bucketed implementation that production systems actually use.
category: Pretraining Concepts · Part 4 of 10
date: 2026-06-11
read_time: 14 min read
---

# Data Parallelism and Gradient Reduction

If you are training a model that fits on one GPU, you can still train it faster by
replicating it across multiple GPUs, each processing a different batch of data, and
averaging the gradients after each backward pass. That is data-parallel training
(DDP), and it is the first parallelism strategy every pretraining setup should have.

This article covers how DDP gradient reduction works, why the naive synchronous
implementation is inefficient, how async reduction introduces overlap, and why
production implementations go one step further and use buckets.

It assumes the primitives from the previous article: ranks, process groups, and
the meaning of collective operations such as all-reduce and reduce-scatter.

---

## The Gradient All-Reduce

In DDP, every GPU holds an identical copy of the model parameters. Each GPU receives
a different batch of data, runs a forward and backward pass, and computes its own
gradient tensor for each parameter.

Before the optimizer step, these gradients must be **averaged across all GPUs** so
that every GPU performs the same parameter update. An optimizer step on unaveraged
gradients would produce different parameters on each GPU — the copies would diverge
and the training would become incoherent.

The operation that computes this average is an **all-reduce**: every rank contributes
its local value, and every rank receives the global average. For gradient reduction,
the all-reduce runs on each parameter's gradient tensor, giving each rank the same
averaged gradient to feed to the optimizer.

```python
# Minimal synchronous DDP (what naive implementations look like)
loss.backward()
for param in model.parameters():
    if param.grad is not None:
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group)
optimizer.step()
```

This works correctly, but has a critical performance problem: **it serializes
communication and computation**. The all-reduce on each parameter runs after the
entire backward pass completes. The GPU is idle while the network is busy.

---

<div class="article-figure">
  <img src="../assets/ddp-gradient-reduction.svg" alt="Three DDP gradient-reduction timelines: post-backward sync, per-parameter async, and bucketed async">
</div>

---

## From Sync to Async to Bucketed

There are three useful mental models here.

The fully synchronous baseline is the simplest: finish backward, then reduce the
gradients. It is easy to reason about, but there is no overlap at all between
communication and computation.

The first improvement is to launch an async all-reduce as soon as each gradient is
ready. This is already much better, because network work can overlap with the rest
of backward. But it still launches one collective per parameter tensor, which is
too many small NCCL operations for large models.

Bucketed DDP is the practical next step. It keeps the overlap idea from async
reduction, but batches many parameter gradients into a flat buffer so each
collective is larger and cheaper to launch.

---

## The Bucketed Async Implementation

The key observation is that we don't need to wait for the full backward pass to
start reducing gradients. Gradients are computed in reverse order during backward —
the later layers get their gradients first. In principle, we could all-reduce each
gradient immediately. In practice, we wait until a whole bucket is ready, then
all-reduce that flat buffer in the background while backward continues on earlier
layers.

The `BucketDataParallelPlugin` implements this with a flat-buffer bucketing scheme:

1. **Before training starts**, parameters are grouped into buckets of ~25 MB each
   (configurable). Each bucket gets a contiguous flat buffer.
2. Each parameter's gradient view points directly into the bucket's flat buffer —
   no copies needed when backward writes into `.grad`.
3. **A hook fires** when the last parameter in a bucket completes its backward pass.
   The hook launches an async `all_reduce` on the bucket's flat buffer.
4. While the async all-reduce runs on the network, backward continues on the
   remaining earlier layers.
5. **At `POST_BACKWARD`**, all bucket handles are waited on before the optimizer step.

The async all-reduce runs on a separate CUDA stream from the backward pass,
so the two operations execute in parallel on the GPU. NCCL manages its own stream
internally; the `handle.wait()` at step 5 inserts a stream dependency that blocks
the optimizer step until all communication has completed.

```python
# Core idea: when the last grad in a bucket is ready, launch async all-reduce
def make_hook(self):
    def hook(param):
        self.pending -= 1
        if self.pending == 0:
            self.handle = dist.all_reduce(
                self.flat_buffer,
                op=dist.ReduceOp.AVG,
                group=self.group,
                async_op=True,
            )
    return hook
```

Then each parameter in the bucket registers that hook:

```python
# Pseudocode: attach the same bucket hook to every param in the bucket
hook = bucket.make_hook()
for param in bucket.params:
    param.register_post_accumulate_grad_hook(hook)
```

`register_post_accumulate_grad_hook()` is what makes the overlap possible. PyTorch
calls it immediately after a parameter accumulates its gradient during backward,
so the bucket can observe gradients becoming ready in backward order instead of
waiting for the entire backward pass to finish.

---

## Why Flat Buffers

The bucket's flat buffer is the implementation detail that makes bucketed DDP
efficient. When multiple parameters share a single contiguous buffer, they can be
all-reduced in a **single NCCL call** rather than one call per parameter.

Just as important, flattening does **not** mean gradients get copied into another
buffer after backward. The bucket wires each `param.grad` directly onto a view of
the flat buffer:

```python
view = flat_buffer[offset : offset + param.numel()].view_as(param)
param.grad = view
```

So when backward writes into `param.grad`, it is already writing into the buffer
that will be all-reduced later. The flat buffer reduces NCCL launch overhead
without adding an extra gradient copy on the hot path.

NCCL (NVIDIA's Collective Communications Library) has a fixed latency cost per
operation. Without bucketing, each parameter *tensor* (weight matrix, bias vector,
etc.) would require its own all-reduce call. A GPT-style model has on the order of
thousands of parameter tensors — one all-reduce per tensor means thousands of small
NCCL calls, each incurring latency overhead.

Bucketing reduces this to a handful of large calls. For a 7B model with bf16
gradients (2 bytes/param), grouping into 25 MB buckets gives:

```
n_calls = total_gradient_bytes / bucket_size
         = (7B params × 2 bytes/param) / (25 × 1024 × 1024 bytes)
         ≈ 533 calls
```

Roughly 500 large NCCL operations instead of thousands of tiny ones. The latency
savings compound: less per-operation overhead, better NIC pipelining, and the
async path can hide most of the cost entirely.

---

## Gradient Accumulation Under DDP

With gradient accumulation (multiple micro-steps per optimizer step), the all-reduce
should only happen on the last micro-step. All-reducing on every micro-step wastes
network bandwidth and would double-count the normalization.

The simplest way to think about this is as a two-state bucket:

- On non-boundary micro-steps, set `pending = 0`.
- On the final micro-step of the accumulation window, set `pending = len(bucket.params)`.

That is enough to control whether the backward hooks merely accumulate gradients
or actually trigger communication:

```python
# Pseudocode
if is_last_microstep:
    bucket.pending = len(bucket.params)   # arm the bucket
else:
    bucket.pending = 0                    # accumulate only

def hook(param):
    bucket.pending -= 1
    if bucket.pending == 0:
        launch_async_all_reduce(bucket.flat_buffer)
```

Before the accumulation boundary, hooks still fire, but the bucket is not armed,
so no all-reduce is launched. Gradients just keep accumulating into the flat
buffer across micro-steps.

At the final boundary micro-step, the bucket is armed with `len(bucket.params)`.
Now each parameter hook counts down toward zero, and when the last gradient in the
bucket arrives, the async all-reduce launches exactly once.

---

## Why Expert Parameters Need Special Handling

Mixture-of-Experts makes gradient synchronization trickier because not every rank
holds the same expert weights.

For ordinary shared parameters, DDP's rule is simple: every DP rank has the same
parameter, so their gradients should be averaged over the DP group.

Expert parameters break that assumption. Along the EP dimension, ranks usually own
different experts, so those weights are **sharded**, not replicated. Averaging such
gradients across the full DP group would mix gradients from different experts,
which is mathematically wrong.

At the same time, many real layouts still have **replicas of the same expert** on
some other axis. In practice, systems often need a separate **expert-replica
group**: ranks in that group hold the same expert weights and should average their
expert gradients together, even though different experts are still distributed
across the EP dimension.

That means expert parameters need a different question from shared parameters:

- If this parameter is a shared weight, reduce it over the DP group.
- If this parameter is an expert weight with replica degree 1, do not reduce it at all.
- If this parameter is an expert weight with multiple replicas, reduce it only across the ranks that hold the **same expert copy**.

So the real rule is not "all expert params skip DDP" or "all expert params reduce
in EP." The real rule is: **reduce each parameter only across its true replica
group**. Shared weights and expert weights often have different replica groups,
which is why DDP has to treat expert parameters carefully.

---

## What's Next

The next article covers ZeRO optimizer sharding: how the optimizer state, gradients,
and eventually the parameters themselves can be sharded across data-parallel ranks
to cut memory, what the three ZeRO stages look like under the hood, and where each
stage's extra communication comes from.

DDP scales throughput but leaves every rank holding a full copy of the training
state. ZeRO attacks that redundancy directly — it stays on the data-parallel axis
but stops replicating what doesn't need to be replicated, the natural next step
once memory, not throughput, is the binding constraint.
