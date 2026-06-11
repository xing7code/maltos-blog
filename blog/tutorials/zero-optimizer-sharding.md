---
layout: post
title: "ZeRO Optimizer Sharding"
description: "Adam's optimizer state is three times larger than the model. ZeRO-1, 2, and 3 progressively shard it across data-parallel ranks — with ZeRO-3 sharding the parameters themselves. This article covers how each stage works, why ZeRO-3 requires module-level all-gathers, and what it costs in communication."
category: Tutorial · Part 5 of 5
date: 2026-06-11
read_time: 16 min read
---

# ZeRO Optimizer Sharding

Training a 7B-parameter language model with AdamW requires storing, for each
parameter, the parameter itself, its gradient, and Adam's first and second moment
estimates (m and v vectors). In mixed-precision training:

| State | Dtype | Bytes/param |
|---|---|---|
| Parameters | bf16 | 2 |
| Gradients | bf16 | 2 |
| Adam m (first moment) | fp32 | 4 |
| Adam v (second moment) | fp32 | 4 |
| **Total** | | **12** |

For 7B parameters: 12 × 7 × 10⁹ ≈ 84 GB. A single H100 has 80 GB. Single-GPU
training is impossible.

ZeRO (Zero Redundancy Optimizer) distributes this memory across data-parallel
ranks. All `N` ranks hold the full batch of input tokens (actually different
batches — they're DP replicas), so some state must be replicated. But the question
is *how much* must be replicated, and the answer is: much less than you'd think.

---

<div class="article-figure">
  <img src="../assets/zero-stages-diagram.svg" alt="ZeRO stages comparison: memory per rank and communication cost">
</div>

---

## Stage 1: Shard the Optimizer State

The insight behind ZeRO-1: **all DP replicas perform the same parameter update,
but each replica only needs to update a subset of parameters at any given time**.

In ZeRO-1, we partition the parameters across DP ranks. Rank `i` is responsible for
parameters `i/N` through `(i+1)/N` of the full parameter set. After the all-reduce,
rank `i` only updates its assigned parameters, discards the gradients for the others,
and all-gathers the updated parameters so all ranks stay in sync.

```
After all-reduce:
  Rank 0 holds: averaged grads for all params
  Rank 0 updates: params 0..N/4 (its shard)
  Rank 0 discards: grads for N/4..N
  
All-gather restores full param set on all ranks.
```

The memory saving: optimizer state (m, v) is sharded by `N`. Each rank stores full
parameters and full gradients, but only `1/N` of the optimizer tensors.

After the optimizer step, each rank has updated its assigned parameter shard —
but the parameters on different ranks are now inconsistent: rank 0 updated params
0..N/4 using its local moments, rank 1 updated params N/4..N/2, etc. A final
all-gather broadcasts each rank's updated shard to all other ranks, restoring
a consistent full parameter copy on every rank.

For a 7B model at N=8:
- Before ZeRO: 84 GB per rank
- After ZeRO-1: 2B (params) + 2B (grads) + (4B+4B)/8 (moments) = 5B bytes/param ≈ 35 GB per rank

---

## Stage 2: Also Shard the Gradients

ZeRO-2 shards both optimizer state and gradients. After backward, instead of
all-reducing the full gradient tensor, we reduce-scatter:

```
All-reduce (DDP/ZeRO-1):
  all ranks → average of all grads → all ranks  (every rank receives the full averaged grad)

Reduce-scatter (ZeRO-2/3):
  all ranks → sum of grads → each rank receives only its assigned shard of the averaged grad
```

All-reduce and reduce-scatter move the same total bytes across the network —
the bandwidth cost is identical. The difference is what ends up in memory: with
all-reduce, every rank stores the full averaged gradient tensor. With reduce-scatter,
rank `i` only stores `1/N` of it — the shard for its assigned parameters. The other
ranks' gradient data is never written to this rank's memory.

The result: after reduce-scatter, only the owning rank needs to keep the gradient.
Other ranks can free their portion immediately, cutting gradient memory by `N`.

ZeRO-2 memory per rank:
- 2B bytes/param for parameters (still replicated — every rank needs full params for forward pass)
- (2B + 4B + 4B)/8 = 1.25B bytes/param for grad + moments
- Total: ~3.25B bytes/param × 7B = ~23 GB per rank (vs. 84 GB with no ZeRO)

---

## Stage 3: Also Shard the Parameters

ZeRO-3 goes all the way: parameters, gradients, and optimizer state are all sharded.
Each rank stores only `1/N` of the actual parameters.

This creates a problem: **each forward pass needs the full parameters for each layer**.
A layer whose weights are sharded across 8 ranks can't compute a matrix multiply.

ZeRO-3 solves this with **just-in-time parameter materialization**: before a layer's
forward pass, all ranks all-gather the full parameter shard. The layer runs. Then
the parameters are immediately freed.

```
Forward pass for layer L:
  1. All-gather W_L from all ranks → W_L_full (on all ranks)
  2. Run forward pass: Y = X @ W_L_full
  3. Free W_L_full (keep only W_L_shard)

Backward pass for layer L:
  1. All-gather W_L from all ranks → W_L_full (needed for gradient computation)
  2. Compute dL/dX and dL/dW_L_full
  3. Reduce-scatter dL/dW_L_full → each rank keeps its shard of the gradient
  4. Free W_L_full
```

ZeRO-3 memory per rank: (2B + 2B + 4B + 4B)/8 = 1.5B bytes/param × 7B ≈ **10 GB per rank**.

---

## Module-Level Wrapping in MALTOS

MALTOS's `Zero3Plugin` wraps modules at the `nn.Module` boundary (default: `nn.Linear`)
rather than at individual parameter boundaries. This is a critical implementation
choice.

**Why modules, not parameters?** All-gathering one parameter at a time would mean
one NCCL call per parameter per forward pass. A 7B model with ~500 million
`nn.Linear` weight matrices would require thousands of all-gathers per step.

By wrapping at the module level, all parameters belonging to one `nn.Linear` are
collected into a single flat buffer (a "bucket") and handled with a single all-gather:

```python
# From zero3.py: wrapping happens in _prepare_buckets
for module_name, module in model.named_modules():
    if not isinstance(module, tuple(self.wrap_cls)):
        continue
    params = [param for param in module.parameters(recurse=True)]
    # All params in this module become one bucket
    bucket = self._make_bucket(index, module, params, logical_names)
```

Each bucket has a local shard `bucket.local_param` (the `1/N` fraction that this
rank owns) and a full data buffer `bucket.exec_state.data_buffer` (populated on
demand during forward/backward).

---

## Prefetching: Hiding All-Gather Latency

A naive ZeRO-3 would: gather → compute → free → gather → compute → free, with no
overlap. The gather latency stalls the forward pass.

MALTOS prefetches the next bucket's parameters while the current bucket is computing:

```python
def _make_materialize_forward_hook(self, bucket):
    def hook(_module, _inputs):
        self._record_forward_bucket(bucket)
        self._materialize_full_params(bucket, direction=FORWARD)
        # While this layer computes, start gathering the NEXT layer's params
        if self.bucket_order_checked and bucket.next_bucket is not None:
            self._prefetch_bucket(bucket.next_bucket, direction=FORWARD)
    return hook
```

On the first step, the bucket execution order is unknown (it depends on the model's
forward pass order). MALTOS records the order dynamically:

```python
def _record_forward_bucket(self, bucket):
    if self.bucket_order_checked or bucket.index in self._observed_forward_set:
        return
    self._observed_forward_set.add(bucket.index)
    self._observed_forward_order.append(bucket)
```

Once `len(observed) == len(buckets)`, `bucket_order_checked = True` and the
`next_bucket` / `prev_bucket` links are established. From step 2 onwards, each
bucket knows exactly which bucket to prefetch next.

In the backward pass, the same logic applies in reverse: as each bucket computes
its backward, it prefetches the previous bucket's parameters (backward runs in
reverse forward order).

**What if the forward pass order is non-deterministic?** Models with dynamic
control flow (e.g., conditional expert routing in MoE, or Python `if` branches
on input content) may visit modules in a different order on different steps. If
this happens, the recorded `next_bucket` links become stale — the prefetch fires
for the wrong bucket. MALTOS handles this conservatively: once `bucket_order_checked`
is set, it stays set. If the actual order differs from the recorded order, the
prefetch misses and the all-gather happens on demand rather than ahead of time.
Correctness is preserved; only the latency benefit is lost. Models with highly
variable execution paths may see reduced prefetch efficiency.

---

## The Reduce-Scatter in Detail

After backward, each parameter in the bucket has accumulated a gradient in
`state.grad_buffer`. The reduce-scatter averages these gradients and delivers
rank `i`'s shard to rank `i`:

```python
def _reduce_scatter_avg(self, bucket, state):
    work = dist.reduce_scatter_tensor(
        state.shard_buffer,       # output: rank i's gradient shard
        state.grad_buffer,        # input: full concatenated gradient
        op=dist.ReduceOp.AVG,
        group=bucket.group_context.group,
        async_op=True,
    )
    return ReduceScatterShardWork(work, state.shard_buffer, bucket.local_param.grad, ...)
```

The reduce-scatter runs async. The `ReduceScatterShardWork` wrapper accumulates
the shard into `bucket.local_param.grad` when `.wait()` is called at `PRE_STEP`.

The optimizer then runs `optimizer.step()` on `bucket.local_param` — the parameter
shard — using the accumulated gradient shard. Each rank updates only its own
parameters.

---

## The Optimizer Factory Pattern

Because ZeRO-3 replaces each parameter's `param.data` with a shard, the optimizer
must be created **after** all parameter replacements are done. If you create the
optimizer at plugin init time, it holds references to the original, pre-sharding
parameters. After ZeRO-3's `transform_model()` runs, those original tensors are
detached from the computation graph, and all optimizer updates go nowhere — silently.

The factory pattern solves this:

```python
def transform_model(self, model: nn.Module) -> nn.Module:
    # ... wrap all modules into buckets, create local_param shards ...
    self._prepare_buckets(model)  # each bucket now has a local_param shard tensor
    
    optimizer_params = [bucket.local_param for bucket in self.buckets]
    self.optimizer = self.runtime.create_optimizer(optimizer_params)
    # ↑ called AFTER all sharding, with the final shard Parameter objects
    self.scheduler = self.runtime.create_scheduler(self.optimizer)
    return model
```

`runtime.create_optimizer()` invokes the user-provided optimizer factory with the
final parameter list. The optimizer's internal state tensors (the `m` and `v`
moment vectors) are created to match the size of the parameters it receives. If the
optimizer were created before `_prepare_buckets`, it would receive the original
full-size parameter tensors — and its `m`/`v` vectors would be sized for the full
parameters, not the shards. After `_prepare_buckets` creates new `local_param`
tensors, those old optimizer entries point to detached tensors that are no longer
updated during training. The silently broken result: optimizer state is computed but
applied to the wrong memory locations.

The runtime doesn't know whether the parameters are full or sharded — the factory
receives whatever the final `local_param` list is.

---

## What `override_param_state_dict` Does

Without ZeRO-3, the default checkpoint saves `model.state_dict()`, which returns
the current value of each named parameter. In a ZeRO-3 run, each parameter's
`.data` is a 1/N shard, not the full weight. A naive `model.state_dict()` would
save those shards with the original parameter names but wrong shapes — loading
them on a different world size (or even the same one) would fail a shape check.

ZeRO-3 overrides this to save shards explicitly with their layout recorded:

```python
def override_param_state_dict(self):
    state = {}
    metadata = []
    for bucket in self.buckets:
        state_key = f"zero3_bucket_{bucket.index}"
        state[state_key] = bucket.local_param.detach().cpu().clone()
        metadata.append(ParamState(
            state_key=state_key,
            logical_names=bucket.logical_names,  # which original params are in this shard
            ...
        ))
    return state, metadata
```

Each rank saves its own shard under a key like `zero3_bucket_0`. The checkpoint
manifest records the world size and per-rank shard extents. On load, the
corresponding shard is read back into `bucket.local_param.data`, and the forward
hooks re-establish on-demand materialization from that point.

The manifest records enough information so that a future implementation can detect
a world-size change on resume and reshard accordingly — though that's not yet
implemented.

---

## ZeRO Stages at a Glance

| Stage | What is sharded | Memory/rank (7B model, N=8) | Extra communication vs. DDP |
|---|---|---|---|
| DDP (no ZeRO) | Nothing | ~84 GB | baseline (1× all-reduce per step) |
| ZeRO-1 | Optimizer state (m, v) | ~35 GB | +all-gather after optimizer step |
| ZeRO-2 | Optimizer state + gradients | ~23 GB | reduce-scatter replaces all-reduce |
| ZeRO-3 | Optimizer state + gradients + parameters | ~10 GB | +all-gather on every forward + backward |

The memory numbers assume mixed-precision training (bf16 params/grads, fp32 moments).
ZeRO-3's ~10 GB per rank is under what a single H100 holds; ZeRO-2 and ZeRO-1
still require multiple GPUs or NVLink for the all-reduce to remain efficient.

---

## Communication Volume: Is ZeRO-3 Worth It?

Per optimizer step, ZeRO-3 costs:
- **Forward pass**: all-gather for each bucket (2× model size, once per forward step,
  but spread across the step via prefetch)
- **Backward pass**: all-gather for each bucket again (another 2× model size)
- **Gradient sync**: reduce-scatter (1× model size)
- **Total**: ~3× model size per step in communication volume

Compare to plain DDP all-reduce: 2× model size per step.

ZeRO-3 costs 50% more communication per step, but uses `1/N` the memory for
parameters, gradients, and optimizer state. At N=8, that's 12 GB vs. 84 GB for
a 7B model.

The practical answer: ZeRO-3 is worth it when the model doesn't fit without it,
and when you have enough bandwidth to hide the communication. On NVLink-connected
A100/H100 nodes, the bandwidth is high enough that prefetch largely hides the
all-gather latency. Across nodes on InfiniBand, the communication starts to dominate.

---

## Gradient Accumulation Under ZeRO-3

ZeRO-3 with gradient accumulation requires careful state management. On each
micro-step, ZeRO-3:

1. Materializes parameters (forward all-gather)
2. Runs backward, accumulating gradients into `state.grad_buffer`
3. On the last micro-step only: runs the reduce-scatter

The `exec_states` list has one entry per micro-step. Each micro-step has its own
`grad_buffer` to accumulate into, preventing gradient contributions from different
micro-steps from clobbering each other. The reduce-scatter runs once per optimizer
step (not per micro-step) — it fires on the last micro-step when `is_step_boundary`
is true, exactly like the DDP hook in Part 3.

---

## Experiment Placeholder

> **[Placeholder: ZeRO-3 vs. ZeRO-1 throughput and memory on 8 GPU]**
> Benchmark `Zero3Plugin` vs. `Zero1Plugin` on a 3B model at dp=8.
> Expected: ZeRO-3 significantly lower peak memory (by ~3×), lower
> throughput (due to extra all-gathers) unless prefetch hides the latency.
> Measure: tokens/sec, `torch.cuda.max_memory_allocated()`, TFLOPS efficiency.
> Compare with and without prefetch (can be toggled by setting `bucket_order_checked=True`
> prematurely to test without the prefetch path).

---

## What's Next in This Series

This concludes the tutorial series. You now have the building blocks:

1. **The training loop** — gradient accumulation, mixed precision, checkpointing
2. **Data storage** — token shards, memory-mapped access, DP-aware cursors
3. **Data parallelism** — all-reduce, bucketed DDP, communication-compute overlap
4. **Tensor and sequence parallelism** — column/row sharding, SP activation sharding
5. **ZeRO optimizer sharding** — stages 1/2/3, factory pattern, prefetch

These five techniques cover the space of what most production pretraining jobs use.
The deep-dive series goes into the implementation decisions that make composing these
techniques non-trivial: how the plugin system enforces ordering, how the optimizer
factory prevents silent failures, how checkpoints stay correct under all combinations,
and what pipeline parallel schedules actually look like at the code level.
