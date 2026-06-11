---
layout: post
title: "Data Parallelism and Gradient Reduction"
description: DDP is the first parallelism strategy every pretraining setup needs. This article covers synchronous and asynchronous gradient all-reduce, the bucketed implementation that overlaps communication with backward, and why the naive approach leaves performance on the table.
category: Tutorial · Part 3 of 5
date: 2026-06-11
read_time: 14 min read
---

# Data Parallelism and Gradient Reduction

If you are training a model that fits on one GPU, you can still train it faster by
replicating it across multiple GPUs, each processing a different batch of data, and
averaging the gradients after each backward pass. That is data-parallel training
(DDP), and it is the first parallelism strategy every pretraining setup should have.

This article covers how DDP gradient reduction works, why the naive synchronous
implementation is inefficient, and how the bucketed approach achieves nearly full
overlap between computation and communication.

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
  <img src="assets/ddp-gradient-reduction.svg" alt="DDP gradient reduction: naive vs. bucketed">
</div>

---

## The Bucketed Approach: Overlap Communication with Backward

The key observation is that we don't need to wait for the full backward pass to
start reducing gradients. Gradients are computed in reverse order during backward —
the later layers get their gradients first. As soon as a layer's gradient is ready,
we can start all-reducing it in the background while backward continues on earlier
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

```python
# From ddp.py: the hook that fires when a bucket is fully ready
def make_hook(self):
    is_gloo = dist.get_backend(self.group) == "gloo"
    world_size = dist.get_world_size(self.group)

    def hook(param):
        self.pending -= 1
        if self.pending == 0:
            self.handle = dist.all_reduce(
                self.flat_buffer,
                op=dist.ReduceOp.AVG,
                group=self.group,
                async_op=True,  # key: non-blocking
            )
    return hook
```

The hook fires via `register_post_accumulate_grad_hook()`, which PyTorch calls
immediately after each parameter accumulates its gradient during backward. This is
what makes the overlap possible: the hook fires as soon as the parameter is ready,
not after all parameters are ready.

---

## Why Flat Buffers

The bucket's flat buffer is the implementation detail that makes bucketed DDP
efficient. When multiple parameters share a single contiguous buffer, they can be
all-reduced in a **single NCCL call** rather than one call per parameter.

NCCL (NVIDIA's Collective Communications Library) has a fixed latency cost per
operation. A model with 10,000 parameters would require 10,000 separate all-reduce
calls if each parameter were reduced independently. Grouping them into 25 MB buckets
reduces this to roughly:

```
n_calls = total_param_bytes / bucket_size
         = (7B params × 4 bytes/param) / (25 × 1024 × 1024 bytes)
         ≈ 1,066 calls
```

For a 7B model at 4 bytes/param, that's ~1,000 NCCL operations instead of 7 billion.
The latency savings compound: less overhead per operation, better pipelining.

The `finalize()` method sets up these views when the bucket is first created:

```python
def finalize(self) -> None:
    total_numel = sum(param.numel() for param in self.params)
    self.flat_buffer = torch.zeros(total_numel, dtype=..., device=...)
    offset = 0
    for param in self.params:
        view = self.flat_buffer[offset : offset + param.numel()].view_as(param)
        self.param_views.append(view)
        offset += param.numel()
    # Assign param.grad to point into the flat buffer
    for param, view in zip(self.params, self.param_views):
        param.grad = view
```

Now `param.grad` is not a separately allocated tensor — it is a view into the
bucket's flat buffer. When backward writes into `param.grad`, it writes directly
into the buffer that will be all-reduced. Zero copies.

---

## Gradient Accumulation Under DDP

With gradient accumulation (multiple micro-steps per optimizer step), the all-reduce
should only happen on the last micro-step. All-reducing on every micro-step wastes
network bandwidth and would double-count the normalization.

The `BucketDataParallelPlugin` handles this with the `is_step_boundary` flag:

```python
def on_phase(self, phase: RuntimePhase) -> None:
    if phase == RuntimePhase.PRE_BACKWARD:
        context = self.runtime.state.step_context
        should_sync = context.is_step_boundary  # True only on last micro-step
        accum_start = context.accum_start       # True only on first micro-step
        for bucket in self.buckets:
            bucket.reset(
                grad_accum_start=accum_start,
                grad_accum_end=should_sync,
            )
```

On micro-steps 1 through N-1, `should_sync=False`. The `bucket.reset()` sets
`bucket.pending = 0` (not `len(self.params)`), so the hook never fires and no
all-reduce is launched. Gradients accumulate in the flat buffer unmolested.

On micro-step N, `should_sync=True`. The `bucket.reset()` sets `pending =
len(self.params)`, arming the hook. As each parameter completes its backward pass
on the last micro-step, the bucket's pending count decrements, and when it reaches
zero, the all-reduce launches.

---

## Expert Parameters and DP Group Exclusion

Not all parameters should be reduced over the DP group. Expert parameters in a
Mixture-of-Experts model are already handled by the Expert Parallelism plugin with
a separate EP all-reduce. Reducing them again over the DP group would double-count.

The DDP plugin checks the parameter role:

```python
for name, param in self.runtime.model.named_parameters():
    if not param.requires_grad:
        continue
    if self.runtime.get_param_role(param) == ParamRole.EXPERT:
        continue  # skip: EP plugin handles these
    # ... add to all-reduce
```

The bucket builder has the same exclusion:

```python
[p for p in model.parameters()
 if p.requires_grad
 and self.runtime.get_param_role(p) != ParamRole.EXPERT]
```

This works because MALTOS tracks each parameter's role via `register_param_role()`,
which EP plugins call during `transform_model`. By the time DDP is set up (which
also happens during `transform_model`, but runs after EP due to the plugin ordering),
the role annotations are already present.

---

## Gloo vs. NCCL

The all-reduce operation behaves differently depending on whether the process group
uses the Gloo or NCCL backend. NCCL runs on CUDA, supports `ReduceOp.AVG` directly,
and can run asynchronously. Gloo runs on CPU and requires manual normalization:

```python
if is_gloo:
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
    param.grad.div_(world_size)
else:
    handle = dist.all_reduce(
        param.grad, op=dist.ReduceOp.AVG, group=dp_group, async_op=True
    )
```

Gloo is slower and can't overlap with backward. It is used for CPU smoke tests —
the same code path as the GPU runs, just without the async benefits. This is how
all DDP combinations are validated before running on real hardware.

---

## How DDP Interacts With ZeRO

When ZeRO-3 is in the configuration, DDP is not loaded. ZeRO-3 handles gradient
reduction itself — using a reduce-scatter instead of an all-reduce, which shards
the averaged gradient across DP ranks rather than replicating it. This is one of
the five interaction surfaces: two plugins that both want to own the gradient
reduction step.

The plugin system resolves this through `runs_after` ordering: the ZeRO-3 plugin
declares `runs_after={PluginId.DP}`. In practice, ZeRO-3 replaces DDP entirely
in a valid configuration — you pick one or the other. Trying to use both would
conflict on the `PluginId.DP` slot, which is enforced by the plugin ID uniqueness
constraint.

---

## The `runtime_optimizer_replicated_axes` Method

After gradient reduction, the optimizer state needs to know whether parameters are
replicated across certain axes. DDP declares that parameters are replicated along
the DP axis:

```python
def runtime_optimizer_replicated_axes(self) -> set[MeshAxis]:
    return {MeshAxis.DP} if self.runtime.mesh.dp > 1 else set()
```

This information is used when the checkpoint system decides how to save optimizer
state. If parameters are replicated, only one rank per DP group needs to save the
optimizer state for that parameter — all replicas are identical. The manifest's
`optimizer_source_ranks` field records which rank to load from.

---

## Scaling Beyond a Single Machine

All of the above works on a single machine with multiple GPUs connected by NVLink.
On a multi-node setup, the DP group spans machines connected by InfiniBand or
RoCE. The code is identical — PyTorch's distributed package abstracts the transport.
The difference is performance: NVLink bandwidth (~600 GB/s) is an order of magnitude
higher than InfiniBand (~200 GB/s), so the relative cost of DDP all-reduce increases
as you span more machines.

This is why large-scale pretraining combines DDP with ZeRO: ZeRO-3 reduces the
communication volume from 2× model size (all-reduce) to 3× model size (all-gather
+ reduce-scatter), but the per-step communication is spread across parameter fetches
rather than one big synchronization point. Combined with TP that reduces the model
size visible at each DP rank, the communication overhead becomes manageable at
hundreds of GPUs.

---

## Experiment Placeholder

> **[Placeholder: DDP vs. BucketDDP throughput comparison]**
> A useful experiment here: benchmark `DataParallelPlugin` (sync) vs.
> `BucketDataParallelPlugin` (async bucketed) on a 1B model with dp=4 on the same
> machine. Expected result: bucketed gets closer to peak GPU utilization (higher
> MFU) by overlapping the ~400ms all-reduce with the backward pass. At dp=2 on
> NVLink the gap may be small; at dp=8 spanning nodes on InfiniBand it should be
> significant.

---

## What's Next

The next article covers tensor parallelism and sequence parallelism: how a single
weight matrix is sharded across multiple GPUs, what ColumnParallelLinear and
RowParallelLinear actually do, and how sequence parallelism shards the activations
between layers to cut memory further.

Data parallelism scales training throughput by processing more data in parallel.
Tensor parallelism scales training throughput by making each forward pass faster —
a different axis of parallelism that can be combined with DDP to scale both ways at once.
