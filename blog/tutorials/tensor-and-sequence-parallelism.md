---
layout: post
title: "Tensor and Sequence Parallelism"
description: TP shards each weight matrix across multiple GPUs. SP shards the activations between layers. Together they reduce memory and increase compute throughput — but they require a specific communication pattern that must be woven into the forward and backward passes of every layer.
category: Tutorial · Part 4 of 5
date: 2026-06-11
read_time: 15 min read
---

# Tensor and Sequence Parallelism

Data parallelism replicates the model and distributes the data. Tensor parallelism
does the opposite: it replicates the data (input activations) and distributes the
model (weight matrices). Each GPU holds a slice of the weight matrix, performs a
partial matrix multiply, and the results are combined with a collective communication
operation.

Sequence parallelism extends this by sharding the activations between layers across
the sequence dimension, cutting activation memory by a factor of the TP world size.

Together, TP and SP are the workhorses of large-model training. Data parallelism
alone doesn't solve the memory problem — each DDP replica still holds the full
model. A 70B-parameter model at 12 bytes/param needs ~840 GB; no single GPU has
that. TP splits the model itself across GPUs, so each device only holds `1/N` of
each weight matrix while remaining fully busy with computation.

---

## Why You Need Both Column and Row Sharding

Consider a single weight matrix `W` of shape `[d_in, d_out]`. The computation is
`Y = X @ W`. There are two natural ways to shard `W` across `N` devices:

**Column sharding** (shard along `d_out`): split `W` into `N` column slices.
Each device holds `W_i` of shape `[d_in, d_out/N]`. Each device computes
`Y_i = X @ W_i`, giving a partial output of shape `[B, T, d_out/N]`. The full
output `Y = concat(Y_0, ..., Y_{N-1})` along the last dimension.

**Row sharding** (shard along `d_in`): split `W` into `N` row slices.
Each device holds `W_i` of shape `[d_in/N, d_out]`. Each device also takes a
slice of `X` along its feature dimension: `X_i = X[:, :, i*d_in/N:(i+1)*d_in/N]`.
Then `Z_i = X_i @ W_i` is a partial result of the correct output shape
`[B, T, d_out]`, but only a fraction of the correct value. The full output
is `Z = sum(Z_0, ..., Z_{N-1})`, which requires an all-reduce.

A transformer MLP block is two linear layers: `FFN(x) = GELU(x @ W_1) @ W_2`.
The canonical TP sharding is:
- `W_1`: column sharding → `ColumnParallelLinear`
- `W_2`: row sharding → `RowParallelLinear`

This pairing avoids an all-reduce between the two layers: the column-parallel layer
produces sharded outputs that are exactly the right inputs for the row-parallel layer.
Only one all-reduce is needed, at the output of `W_2`.

---

<div class="article-figure">
  <img src="../assets/tp-sharding-diagram.svg" alt="TP sharding: ColumnParallelLinear and RowParallelLinear">
</div>

---

## ColumnParallelLinear

`ColumnParallelLinear` holds columns `[col_start : col_end]` of the original weight
matrix:

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group, gather_output=False):
        self.weight = nn.Parameter(
            torch.zeros(out_features // world_size, in_features)
        )  # local weight: [d_out/N, d_in] (transposed conv layout)
        self.tp_group = tp_group
        self.gather_output = gather_output

    def forward(self, x):
        # x: [B, T, d_in] — replicated on all ranks
        y = F.linear(x, self.weight)  # [B, T, d_out/N] — sharded
        if self.gather_output:
            # all-gather: used when the next layer expects full output
            y = all_gather_along_last_dim(y, self.tp_group)
        return y
```

The input `x` is replicated across all TP ranks. Each rank's local forward pass
is a standard matrix multiply against its local weight slice. No communication is
needed before or during the matmul.

The output is sharded: each rank has `Y_i` of shape `[B, T, d_out/N]`. If the next
layer is `RowParallelLinear`, this is exactly what it expects. If the next layer
expects the full output (for example, at the end of the model), `gather_output=True`
triggers an all-gather.

---

## RowParallelLinear

`RowParallelLinear` holds rows `[row_start : row_end]` of the original weight:

```python
class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group, comm="none"):
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features // world_size)
        )  # local weight: [d_out, d_in/N]
        self.tp_group = tp_group
        self.comm = comm  # "all_reduce" (vanilla TP) or "reduce_scatter" (SP)

    def forward(self, x_sharded):
        # x_sharded: [B, T, d_in/N] — already sharded by ColumnParallelLinear
        z_partial = F.linear(x_sharded, self.weight)  # [B, T, d_out] — partial result
        if self.comm == "all_reduce":
            dist.all_reduce(z_partial, group=self.tp_group)  # sum → replicate
        elif self.comm == "reduce_scatter":
            # SP: sum partial results AND scatter to sequence shards in one op
            # output: [B, T/N, d_out] — sequence-sharded for the SP region
            z_partial = reduce_scatter(z_partial, self.tp_group, dim=1)
        return z_partial
```

Each rank's partial result `Z_i = X_i @ W_i` has the full output shape `[B, T, d_out]`
but only contains the contribution from the `i`-th input slice. Summing all partials
gives the correct output.

With `post_comm="all_reduce"`, this sum is an all-reduce. With `post_comm="all_to_all"`,
this is where sequence parallelism enters — more on this below.

---

## Attention Layer Sharding

The attention block has a similar structure. Query, key, and value projections:

```
Q = X @ W_q     W_q: [d_model, d_head × n_heads]
K = X @ W_k     W_k: [d_model, d_head × n_heads]
V = X @ W_v     W_v: [d_model, d_head × n_heads]
```

All three are column-parallel: each TP rank handles a subset of attention heads.
Rank `i` computes attention between its own Q_i, K_i, V_i slices — no
inter-rank communication needed during attention computation. The output projection
is row-parallel:

```
O = Softmax(QKᵀ/√d) @ V @ W_o     W_o: [d_head × n_heads, d_model]
```

The result is again an all-reduce at the end of the attention block. The pattern is
the same as the MLP: column → computation → row → all-reduce.

In MALTOS, the model specifies this sharding through a `TpSpParallelSpec` that
lists each submodule path and its sharding axis. The `TensorParallelPlugin` reads
the spec during `transform_model` and swaps `nn.Linear` modules in place:

```python
# From tp.py
for rule in spec.rules:
    module = model.get_submodule(rule.module_path)
    if rule.shard_axis == TpSpShardAxis.PARAM_OUT:
        model.set_submodule(rule.module_path,
            ColumnParallelLinear.from_linear(module, self.tp_group, ...))
    elif rule.shard_axis == TpSpShardAxis.PARAM_IN:
        model.set_submodule(rule.module_path,
            RowParallelLinear.from_linear(module, self.tp_group, ...))
```

The original `nn.Linear` weights are sliced along the correct axis and copied into
the local replacement module. The rest of the model is unchanged.

---

## Sequence Parallelism: Sharding the Activations

Vanilla TP replicates activations between layers. In the regions between MLP blocks
and attention blocks — layer norms, dropout, residual connections — the full
activation tensor `[B, T, d_model]` sits replicated on every TP rank. This is
memory wasted.

**Sequence parallelism** shards these activations along the sequence dimension.
Between TP layers, each rank holds `[B, T/N, d_model]` rather than `[B, T, d_model]`.
This cuts activation memory by a factor of the TP world size.

The communication changes at every TP layer boundary:

- **Entering a transformer block** (SP region → full sequence): the
  `SequenceParallelPlugin` registers a forward pre-hook on the attention/MLP
  block as a whole — not on the individual linear layers inside it. The hook
  runs before the block's forward method, all-gathering `[B, T/N, d_model]` into
  `[B, T, d_model]`. The `ColumnParallelLinear` layers inside the block then see
  the full-sequence input and compute normally.

- **Exiting a transformer block** (full sequence → SP region): the final
  `RowParallelLinear` uses `comm="reduce_scatter"` instead of `comm="all_reduce"`.
  Rather than summing the partial results and replicating them on all ranks, it
  reduce-scatters along the sequence dimension: each rank receives the fully-summed
  result for only its assigned sequence slice. Output shape: `[B, T/N, d_model]`.

```
SP region (layer norm, residual, dropout):
  each rank holds [B, T/N, d_model]

  ↓  all-gather (SP plugin hook before attention/MLP)

Inside TP layers:
  ColumnParallel input:  [B, T, d_model]   ← full sequence
  ColumnParallel output: [B, T, d_out/N]   ← feature-sharded
  RowParallel output:    reduce_scatter
                       → [B, T/N, d_model] ← back to sequence-sharded

  ↓  next SP region continues with [B, T/N, d_model]
```

The net effect: layer norm, residual connections, and dropout operate on
`[B, T/N, d_model]` rather than `[B, T, d_model]`, cutting their activation
memory by a factor of N. Total communication bandwidth is unchanged versus vanilla
TP (one all-reduce per layer becomes one all-gather plus one reduce-scatter, same
bytes on the wire), but activations in the SP regions are N× smaller.

---

## Memory Impact of TP+SP

For a single transformer layer:

| Activation | Plain TP (no SP) | TP+SP |
|---|---|---|
| Layer norm input | `[B, T, d_model]` replicated | `[B, T/N, d_model]` sharded |
| After QKV proj | `[B, T, d_model/N]` sharded | `[B, T/N, d_model/N]` sharded |
| Full attention `QKᵀ` | `[B, H/N, T, T]` per rank | `[B, H/N, T, T]` (no change, attn is on local heads) |
| After output proj | `[B, T, d_model]` replicated | `[B, T/N, d_model]` sharded |
| FFN intermediate | `[B, T, d_ffn/N]` sharded | `[B, T/N, d_ffn/N]` sharded |

The SP-sharded activations reduce peak memory. At `tp=4`, a single-layer activation
that would occupy 4 GB now occupies 1 GB. For very long contexts (CP territory), SP
is often the first technique deployed before reaching for context parallelism.

---

## Gradient Semantics Under TP

The parameter gradients are local: each rank computes the gradient for its own
weight slice, and no communication is needed. `dL/dW_i = Xᵀ @ dL/dY_i` — a
local matmul using the local input `X` and local output gradient `dL/dY_i`.

The input gradient `dL/dX` is different — it requires communication. Here's why:

In the forward pass of `RowParallelLinear`, every rank's partial result
`Z_i = X_i @ W_i` is summed via all-reduce to produce the full output
`Z = Z_0 + Z_1 + ... + Z_{N-1}`. In the backward pass, to compute how the loss
changes with respect to the input `X`, we need `dL/dX_i = dL/dZ @ W_iᵀ`. But
`dL/dZ` is the gradient that arrives at the output of `RowParallel` — it's a
replicated tensor (same on all ranks, because the forward all-reduce replicated
the output). So `dL/dX_i = dL/dZ @ W_iᵀ` is a local operation, and the
gradient flows back into `ColumnParallelLinear` as a sharded input gradient.

`ColumnParallelLinear`'s backward then needs to compute `dL/dX` — the full-input
gradient that flows to the previous layer. Since each rank holds a slice of the
output `Y_i`, it computes a partial `dL/dX_partial = dL/dY_i @ W_iᵀ`. Summing
these partials across ranks gives the full input gradient — and that sum is the
all-reduce that appears in `ColumnParallelLinear`'s backward. PyTorch's autograd
inserts this automatically because the forward pass used a distribution of work
across ranks: the backward must undo that distribution.

**In short**: weight gradients in TP are local (no communication). Input gradients
require an all-reduce at each `ColumnParallelLinear` boundary, handled transparently
by autograd.

---

## Checkpoint Handling: Annotating Sharded Params

When a TP run saves a checkpoint, each rank has a different slice of each weight
matrix. The `TensorParallelPlugin.annotate_checkpoint_state()` method records this:

```python
tp_shard = {
    "axis": shard_axis.value,   # "param_out" (column) or "param_in" (row)
    "rank": rank,
    "world_size": world_size,
    "shard_dim": shard_dim,     # which dimension was sharded
    "shard_offset": rank * local_extent,
    "shard_extent": local_extent,
    "logical_shape": logical_shape,  # full original shape before sharding
}
```

On load, this annotation tells the loader how to reconstruct the full weight matrix
from the shards across ranks. Changing the TP world size between a checkpoint save
and load requires resharding — a non-trivial operation not yet supported in MALTOS.

---

## Experiment Placeholder

> **[Placeholder: TP+SP memory scaling vs. model size]**
> A useful comparison: activation memory at a fixed sequence length (seq=4096)
> for a 1B model at tp=1 vs. tp=2 vs. tp=4 vs. tp=4+SP. Expected: activation
> memory scales linearly with seq_len at fixed tp; SP adds another factor of tp
> in the activation term. Measure with `torch.cuda.max_memory_allocated()` after
> a single forward pass.

---

## What's Next

The next article covers ZeRO optimizer sharding: how the optimizer state (Adam's
first and second moment estimates) can be sharded across DP ranks to cut memory,
what the three stages look like under the hood, and where each stage's
communication cost comes from.

TP+SP gives you horizontal parallelism within a node. ZeRO gives you vertical
parallelism within the optimizer. The combination of TP+SP+ZeRO-3 is the standard
configuration for models too large to fit on a single GPU even in a multi-rank
setting.
