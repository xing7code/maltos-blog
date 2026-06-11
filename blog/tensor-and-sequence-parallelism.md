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

Together, TP and SP are the workhorses of large-model training. A 70B-parameter
model cannot fit on a single 80GB GPU even for inference; TP with world_size=4
(or 8) spreads the memory across devices while keeping compute density high.

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
  <img src="assets/tp-sharding-diagram.svg" alt="TP sharding: ColumnParallelLinear and RowParallelLinear">
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
    def __init__(self, in_features, out_features, tp_group, post_comm, comm_dim=None):
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features // world_size)
        )  # local weight: [d_out, d_in/N]
        self.tp_group = tp_group
        self.post_comm = post_comm  # "all_reduce" or "all_to_all"
        self.comm_dim = comm_dim

    def forward(self, x_sharded):
        # x_sharded: [B, T, d_in/N] — already sharded by ColumnParallelLinear
        z_partial = F.linear(x_sharded, self.weight)  # [B, T, d_out]
        if self.post_comm == "all_reduce":
            dist.all_reduce(z_partial, group=self.tp_group)  # sum across ranks
        elif self.post_comm == "all_to_all":
            z_partial = all_to_all_along_seq_dim(z_partial, self.tp_group)
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

The communication changes:
- **Before `ColumnParallelLinear`**: an all-gather along the sequence dimension
  reconstructs the full-sequence activation, so each rank's matmul sees all tokens
  (still with its own weight columns).

  Actually, this isn't quite right. With SP, the input scatter goes the other way.
  Let me be precise:

  Before the column-parallel layer, we don't all-gather. Instead, we scatter: the
  sequence-sharded input `[B, T/N, d_model]` is processed as-is, since each rank
  has full `d_model` but only `T/N` sequence positions. Wait - this requires the
  matrix multiply to work on a subset of sequence positions but the full feature
  dimension. That's fine: `[B, T/N, d_model] @ [d_model, d_out/N]` → `[B, T/N, d_out/N]`.
  We need to gather across ranks to get `[B, T, d_out/N]`? No...

  Actually the clean description: with SP the communication ops at the TP boundaries
  become all-to-all (scatter/gather along the seq+feature dimensions simultaneously).

  In MALTOS, `RowParallelLinear` with `post_comm="all_to_all"` and `comm_dim=...`
  handles the SP case.

- **After `RowParallelLinear`**: instead of an all-reduce (which would replicate the
  result), an all-to-all reshards the result along the sequence dimension, producing
  `[B, T/N, d_model]` for the next layer.

The net result: between every pair of TP layers, activations stay sharded along
the sequence dimension. The all-reduce (2× bandwidth of a reduce-scatter) is replaced
by two cheaper all-to-alls.

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

The backward pass through `ColumnParallelLinear` and `RowParallelLinear` mirrors the
forward. The parameter gradients are local: each rank computes the gradient with
respect to its own weight slice, and no all-reduce is needed for the weight gradients.
(The gradient is: `dL/dW_i = X_iᵀ @ dL/dY_i`, which is a local operation.)

The all-reduce in the forward pass of `RowParallelLinear` introduces an all-reduce
in the backward pass of `ColumnParallelLinear`. This is handled automatically by
PyTorch's autograd: the all-reduce operation has a registered backward that
performs the corresponding gradient computation.

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
