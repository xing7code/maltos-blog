---
layout: post
title: "ZeRO Optimizer Sharding"
description: "Adam's optimizer state is three times larger than the model. ZeRO-1, 2, and 3 progressively shard it across data-parallel ranks — with ZeRO-3 sharding the parameters themselves. This article covers how each stage works, why ZeRO-3 requires module-level all-gathers, and what it costs in communication."
category: Pretraining Concepts · Part 6 of 9
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
ranks. In standard DDP, each rank holds a full copy of all parameters, gradients,
and optimizer state — even though those copies are identical (DDP keeps them in
sync via all-reduce). ZeRO exploits this redundancy: each rank only needs to store
the state it is responsible for updating. The question is *how much* must be
replicated, and the answer is: much less than you'd think.

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
After all-reduce (example, N=4 ranks):
  Rank 0 holds: averaged grads for all params
  Rank 0 updates: params 0..N/4   (its assigned shard, 1/N of total params)
  Rank 1 updates: params N/4..N/2
  Rank 2 updates: params N/2..3N/4
  Rank 3 updates: params 3N/4..N

All-gather restores the full param set on all ranks.
```

The memory saving: optimizer state (m, v) is sharded by `N`. Each rank stores full
parameters and full gradients, but only `1/N` of the optimizer tensors.

ZeRO-1 still performs a full gradient all-reduce (like DDP) — every rank receives
the fully-averaged gradient for all parameters. The difference from DDP is that
each rank then discards the gradients for the parameters it doesn't own, only
updating its assigned slice. This is less memory-efficient than ZeRO-2 (which never
materializes the full gradient on any rank), but ZeRO-1 is simpler to implement and
debug because the all-reduce is identical to DDP — only the optimizer step changes.

After the optimizer step, each rank has updated its assigned parameter shard —
but the parameters on different ranks are now inconsistent: rank 0 updated params
0..N/4 using its local moments, rank 1 updated params N/4..N/2, etc. A final
all-gather broadcasts each rank's updated shard to all other ranks, restoring
a consistent full parameter copy on every rank.

For a 7B model at N=8:
- Before ZeRO: 84 GB per rank
- After ZeRO-1: 2 (params) + 2 (grads) + (4+4)/8 (moments) = 5 bytes/param × 7B params ≈ 35 GB per rank

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
the bandwidth cost is identical. (An all-reduce is equivalent to a reduce-scatter
followed by an all-gather; both phases move `model_size` bytes each. ZeRO-2
drops the all-gather phase — it only runs reduce-scatter — but the reduce-scatter
itself moves the same bytes as an all-reduce because every rank still contributes
its full gradient tensor to the reduction.) The difference is what ends up in
memory: with all-reduce, every rank stores the full averaged gradient tensor. With
reduce-scatter, rank `i` only stores `1/N` of it — the shard for its assigned
parameters. The other ranks' gradient data is never written to this rank's memory.

The result: after reduce-scatter, only the owning rank needs to keep the gradient.
Other ranks can free their portion immediately, cutting gradient memory by `N`.

ZeRO-2 memory per rank:
- 2 bytes/param for parameters (still replicated — every rank needs full params for forward pass)
- (2 + 4 + 4)/8 = 1.25 bytes/param for grad + moments (sharded by N=8)
- Total: ~3.25 bytes/param × 7B params ≈ 23 GB per rank (vs. 84 GB with no ZeRO)

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
  1. All-gather W_L from all ranks → W_L_full (must redo; activations were freed
     after the forward pass, so the weight must be rematerialized for dL/dW computation)
  2. Compute dL/dX and dL/dW_L_full
  3. Reduce-scatter dL/dW_L_full → each rank keeps its shard of the gradient
  4. Free W_L_full
```

ZeRO-3 memory per rank: (2 + 2 + 4 + 4)/8 = 1.5 bytes/param × 7B params ≈ **10.5 GB per rank**.

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

Each bucket has a local shard `bucket.local_param` (an `nn.Parameter` holding
the `1/N` fraction of the weight that this rank owns) and a full data buffer
`bucket.exec_state.data_buffer` (a temporary tensor populated during forward/backward
via all-gather, then freed). After `_prepare_buckets` runs, `bucket.local_param`
becomes the "official" parameter that the optimizer tracks — the original
`module.weight` reference is replaced so `model.parameters()` now yields the
sharded `local_param` objects rather than the original full-size tensors.

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

Because ZeRO-3 replaces each parameter with a shard, the optimizer
must be created **after** all parameter replacements are done. Here's the subtle
issue: if you call `optimizer = Adam(model.parameters())` before `transform_model()`,
Adam stores internal references to each parameter *tensor object*. After ZeRO-3's
`transform_model()` runs, it creates new `local_param` tensors (the shards) and
replaces the module's `.weight` attributes with them. The old tensor objects that
Adam holds are now orphaned — they no longer participate in the forward pass.
Adam will compute and apply updates to those orphaned tensors, not to the
`local_param` shards that the model actually uses. The model trains, the loss
decreases (the shards are updated correctly via ZeRO-3's own update path), but the
optimizer state is wasted — silently. Using the factory pattern avoids this by
calling `optimizer = Adam(bucket.local_param for ...)` after the shards exist.

The factory pattern solves this by creating the optimizer with the final shard
parameter list:

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
final `local_param` list. The runtime doesn't know whether the parameters are full
or sharded — it just calls the factory with whatever list it receives. The optimizer's
`m` and `v` moment vectors are then sized to match the shards, not the original
full parameters.

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
| ZeRO-3 | Optimizer state + gradients + parameters | ~10.5 GB | +all-gather on every forward + backward |

The memory numbers assume mixed-precision training (bf16 params/grads, fp32 moments).
ZeRO-3's ~10.5 GB per rank is under what a single H100 holds; ZeRO-2 and ZeRO-1
still require multiple GPUs or NVLink for the all-reduce to remain efficient.

---

## Communication Volume: Is ZeRO-3 Worth It?

Per optimizer step, ZeRO-3 costs:
- **Forward pass**: all-gather for each bucket (1× model size total, once per
  forward step, spread across the step via prefetch). Each rank contributes its
  `1/N` shard; the gather produces the full model on all ranks temporarily.
- **Backward pass**: all-gather for each bucket again (1× model size)
- **Gradient sync**: reduce-scatter (1× model size)
- **Total**: 3× model size per step in communication volume

The "2× model size" figure sometimes cited for the all-gather phases comes from
ring-allreduce accounting, where each byte travels twice. In ZeRO-3's case, the
all-gather is a straightforward gather: each of N ranks sends `model_size/N` bytes,
so total traffic is `model_size × (N-1)/N` ≈ 1× per gather — two gathers = 2×
for parameters, plus 1× for reduce-scatter = 3× total. This 3× figure is the
common way to cite ZeRO-3 overhead in the literature (e.g., the original ZeRO paper).

Compare to plain DDP all-reduce: 2× model size per step (ring-allreduce: each byte
travels through N-1 reduce steps and N-1 broadcast steps).

ZeRO-3 costs 50% more communication per step, but uses `1/N` the memory for
parameters, gradients, and optimizer state. At N=8, that's ~10 GB vs. ~84 GB for
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
is true, exactly like the DDP hook in Part 4.

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

## What's Next

The next article covers pipeline parallelism: how to split a model vertically
across pipeline stages, why microbatching is necessary to keep all GPUs busy, and
the difference between the AFAB and 1F1B schedules that control memory vs. bubble
tradeoffs.

ZeRO eliminates memory redundancy within the optimizer and parameter tensors. Pipeline
parallelism eliminates the need for any single GPU to hold more than a fraction of
the model's layers — a different kind of memory saving that becomes essential when
models grow too deep for TP to handle alone.
