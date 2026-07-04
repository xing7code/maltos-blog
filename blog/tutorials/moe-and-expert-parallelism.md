---
layout: post
title: "Mixture of Experts and Expert Parallelism"
description: MoE replaces a dense FFN with a pool of expert networks and a router that sends each token to one expert. Expert parallelism distributes those experts across GPUs with an all-to-all dispatch that routes tokens to their experts' ranks — and back.
category: Pretraining Concepts · Part 9 of 10
date: 2026-06-11
read_time: 11 min read
---

# Mixture of Experts and Expert Parallelism

Every transformer block has a feed-forward network (FFN): two linear layers with
a nonlinearity between them. In a 7B dense model, each FFN has roughly
`4 × d_model` hidden units, and the FFN parameters account for ~2/3 of total
parameters.

The observation behind Mixture of Experts: most tokens don't need all of that
capacity. A sentence about biology and a sentence about tax law both flow through
the same FFN weights. What if you had separate FFN "experts" for different types
of content, and only routed each token to the expert best suited for it?

Mixture of Experts (MoE) does exactly this. It replaces the dense FFN with N
expert networks and a learned router. Each token activates exactly one expert
(Top-1 routing) or a small subset (Top-K routing). The model has more parameters
but the same FLOPs per token — more capacity without proportional compute.

---

## The MoE Layer

A standard transformer FFN:
```python
# Dense FFN: every token → same weights
def ffn(x):
    return gelu(x @ W1) @ W2  # x: [B, T, d]; W1: [d, 4d]; W2: [4d, d]
```

An MoE layer with N experts and Top-1 routing:
```python
class Top1MoE(nn.Module):
    def __init__(self, dim, hidden_size, num_experts):
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(dim, hidden_size) for _ in range(num_experts)])

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])  # [B*T, d]
        logits = self.router(flat)         # [B*T, N]
        probs = logits.softmax(dim=-1)
        expert_idx = probs.argmax(dim=-1)  # [B*T] — which expert for each token
        weight = probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)  # routing weight

        out = torch.zeros_like(flat)
        for idx, expert in enumerate(self.experts):
            mask = expert_idx == idx
            if not torch.any(mask):
                continue
            out[mask] = expert(flat[mask]) * weight[mask].unsqueeze(1)
        return out.view(x.shape)
```

Key observations:
- The **router** (`Linear(dim, N)`) is tiny — a single linear projection.
- Each token is processed by exactly one expert — the one with the highest router probability.
- The routing **weight** (`weight = probs.gather(...)`, the softmax probability of the chosen expert)
  scales the expert's output. This is not optional: it keeps the router's probability estimates
  meaningful during training. Without the weight multiplication, the router output would be a binary
  choice with no differentiable signal about *how confident* the router was — the gradient through
  the routing decision would be zero everywhere except at the argmax boundary.
- Experts that receive no tokens for a batch are skipped entirely (`if not torch.any(mask): continue`).

**Parameter count**: the FFN accounts for roughly 2/3 of parameters in a
standard transformer. With N experts replacing the FFN:

```
Dense:  params = P_attn + P_ffn
MoE:    params = P_attn + N × P_ffn
```

For a 7B dense model (≈2.3B attention, ≈4.7B FFN):

- 8 experts: 2.3B + 8×4.7B ≈ **40B parameters**, same per-token compute as dense 7B
- 64 experts: 2.3B + 64×4.7B ≈ **303B parameters**, same per-token compute as dense 7B

More experts = more capacity, same compute. This is the core MoE value
proposition. (Top-2 routing — used in Mixtral — activates 2 experts per token,
doubling the per-token compute but also doubling active capacity.)

---

## Why Naive MoE Doesn't Scale

As shown in the parameter count above, an 8-expert MoE with dense-7B per-token compute
stores ~40B parameters — more than 5× the dense 7B model's footprint. A
64-expert version stores ~303B. This is the point: MoE trades memory for
capacity. As you increase the number of experts, the model becomes richer
(more capacity per FLOP) but also larger in memory. Eventually, no single GPU
can hold all experts, and training becomes impossible without distribution.

The solution is expert parallelism: put different experts on different GPUs.

---

## Expert Parallelism: Sharding Experts Across GPUs

Expert parallelism (EP) gives each GPU ownership of `num_experts / ep_world_size`
experts. With 8 experts across 4 EP ranks:

```
EP rank 0: experts 0, 1
EP rank 1: experts 2, 3
EP rank 2: experts 4, 5
EP rank 3: experts 6, 7
```

The problem: after the router assigns each token to an expert, that token may
need to travel to a different GPU — the one that holds its assigned expert.
This requires cross-GPU communication.

---

## The All-to-All Dispatch

The communication primitive is **all-to-all**: every rank sends a (different)
subset of tokens to every other rank, and receives tokens from every other rank.
Unlike all-reduce (which aggregates data) or all-gather (which replicates it),
all-to-all *redistributes* data — a many-to-many routing operation.

<div class="article-figure">
  <img src="../assets/moe-dispatch.svg" alt="All-to-all regroups tokens by destination: each rank holds tokens bound for every expert-rank; after the all-to-all each rank holds only its own experts' tokens, runs them locally, then a second all-to-all sends results home">
</div>

The full dispatch cycle for one MoE layer:

```python
# The algorithm, with the distributed plumbing left abstract:
def moe_ep_forward(x):
    idx, weight = router(x)                    # 1. pick an expert (+ routing weight) per token
    tokens = sort_by_destination_rank(x, idx)  # 2. group tokens by the rank owning their expert
    recv   = all_to_all(tokens)                # 3. DISPATCH: each group flies to its expert's rank
    out    = local_experts(recv) * weight      # 4. run the experts that live on this rank
    home   = all_to_all(out)                   # 5. RETURN: results fly back to each token's origin
    return unsort(home)                        # 6. restore the original token order
```

Steps 1, 2, 4, and 6 are ordinary local tensor ops. The one part that needs a
real distributed API — and the one place people get stuck — is `all_to_all`,
because a rank cannot receive a variable number of tokens without first knowing
how many are coming. `all_to_all_single` needs its receive buffer **pre-allocated**
at exactly the right size, so every dispatch is really *two* exchanges: a tiny one
to swap counts, then the big one to move the tokens.

```python
import torch.distributed as dist
import torch.distributed.nn as dist_nn        # autograd-aware collectives

# send_counts[j] = how many of this rank's tokens are destined for rank j.
# First a tiny all-to-all, so every rank learns how many tokens it will RECEIVE:
recv_counts = torch.empty_like(send_counts)
dist.all_to_all_single(recv_counts, send_counts, group=ep_group)   # just N ints

# Now the real dispatch — move the tokens, split by those counts:
recv_tokens = torch.empty(int(recv_counts.sum()), d, device=x.device, dtype=x.dtype)
dist_nn.all_to_all_single(
    recv_tokens, send_tokens,
    output_split_sizes=recv_counts.tolist(),   # how many arrive from each rank
    input_split_sizes=send_counts.tolist(),    # how many we send to each rank
    group=ep_group,
)

# ... run local experts on recv_tokens ...

# The return trip is the same call with the send/recv split sizes swapped.
```

Using the **autograd-aware** `dist_nn.all_to_all_single` (rather than plain
`dist`) is what makes the layer trainable: the backward pass retraces the same
routes in reverse *automatically* — gradients of the results travel back to the
expert ranks, and gradients of the dispatched tokens travel home. So the two
forward all-to-alls become **four per MoE layer per training step.** The
communication volume per all-to-all is (token count × hidden_size × dtype_bytes)
— *independent of how many experts there are.*

**Shape walkthrough** (4 ranks, 8 experts; `M` tokens on each rank — i.e. batch ×
sequence — with hidden dim `d`):

```
Before dispatch:
  flat:         [M, d]     — all tokens local on each rank
  send_tokens:  [M, d]     — same tokens, sorted by destination rank
  send_counts:  [4]        — how many tokens this rank sends to each other rank

After dispatch all-to-all:
  recv_tokens:  [M', d]    — tokens from all ranks whose expert lives here
                            (M' varies by rank: a hot expert's rank receives more)

After local expert compute:
  expert_out:   [M', d]    — expert-processed results, same shape as recv_tokens

After return all-to-all (inverse routing):
  home:         [M, d]     — results sorted back by source rank

After unsort:
  out:          [M, d]     — results in original token order
```

Every token starts and ends on its original rank. The two all-to-alls are inverse
operations: the first routes tokens to experts, the second routes results home.

---

## Load Imbalance and Auxiliary Loss

Top-1 routing has a failure mode: the router can learn to send all tokens to
one or two experts, causing those experts to be overloaded while others receive
nothing (expert collapse). Collapsed experts receive no gradient signal and never
improve.

The standard fix is an auxiliary load-balancing loss that penalizes imbalanced
routing:

```python
# Auxiliary loss from the Switch Transformer paper (Fedus et al., 2021).
tokens_per_expert = ...  # per-expert assignment counts (shape [num_experts])
expert_fractions = tokens_per_expert.float() / tokens_per_expert.sum()
router_probs_mean = router_probs.mean(dim=0)  # average router prob per expert
aux_loss = (expert_fractions * router_probs_mean).sum() * num_experts
total_loss = language_model_loss + aux_loss_coefficient * aux_loss
```

The intuition: `expert_fractions` measures how many tokens went to each expert
(discrete, not differentiable), and `router_probs_mean` measures how much
probability mass the router assigns to each expert (differentiable). Their dot
product is small when routing is uniform and large when routing is concentrated.
Multiplying by N (the expert count) normalizes the loss to be scale-invariant
across different N. Minimizing it encourages the router to spread tokens evenly.

A minimal Top-1 MoE without the auxiliary loss is fine for small experiments.
At scale, collapse occurs within a few thousand steps without it. Production MoE
runs (Mixtral, Switch Transformer) add the auxiliary loss with a small
coefficient (typically 0.01–0.1) to maintain balance throughout training.

**Imbalance is also a memory problem.** Under EP, the rank holding a hot expert
receives more tokens than its peers — its `recv_tokens` buffer grows with the
imbalance, and in the worst case (full collapse onto one expert) a single rank
receives *every* token in the batch. The exact-count exchange we saw earlier
routes those tokens correctly but offers no protection against the memory spike.
Production systems bound it with a **capacity factor**: each
expert accepts at most `capacity_factor × (tokens / num_experts)` tokens, and
tokens beyond that are **dropped** — they skip the expert and pass through the
residual connection unchanged. Dropping sounds alarming but is standard practice
(Switch Transformer uses capacity factors of 1.0–1.25); the auxiliary loss keeps
the drop rate low.

---

## Gradient Synchronization: Two Groups

In a training run with both DP and EP, there are two classes of parameters:

- **Shared parameters** (embeddings, attention, layer norms): replicated across
  EP ranks. These receive gradients like standard DP parameters — reduce over
  the DP group.
- **Expert parameters** (the expert FFN weights): each EP rank holds different
  experts, so an all-reduce over the full DP group would be wrong — most ranks
  don't even hold the same expert. Expert gradients reduce only over the set of
  ranks that each hold a *copy* of the same expert.

  Where do those copies come from? Take DP=4 and EP=2: EP rank 0 holds experts
  0–3, EP rank 1 holds experts 4–7. Across the 4 DP replicas, every replica has
  the same EP layout — so there are 4 copies of "experts 0–3", one per DP
  replica. Those 4 ranks all computed gradients for the *same* expert weights
  and must average them together. So expert gradients still reduce along the DP
  axis, but *within a fixed EP position* rather than across the whole world.

```python
# after backward — two parameter classes, two reduction groups
for param in expert_params:   # average across replicas of the SAME expert
    dist.all_reduce(param.grad, op=AVG, group=expert_replica_group)

for param in shared_params:   # standard DP all-reduce
    dist.all_reduce(param.grad, op=AVG, group=dp_group)
```

For this to work, each parameter has to be tagged *expert* or *shared* when the
MoE layer is built, and the plain DP reducer must **skip** the expert parameters
— otherwise they'd be reduced twice, over the wrong group. That bookkeeping is
exactly what a runtime handles for you; the deep dive on
[composable parallelism](../internals/composable-parallelism.html) shows how one
framework wires it through a role-filtered gradient callback.

---

## Memory Impact of Expert Parallelism

Without EP: all N experts on one GPU. Memory cost = N × (dense FFN parameter memory).

With EP across `ep_world_size` ranks: each rank holds `N / ep_world_size` experts.
Memory for experts ÷ by EP world size, paid for with the all-to-all communication.

For a 64-expert model with ep_world_size=8: each rank holds 8 experts — 8× the
FFN parameter memory of the dense model, down from 64× without EP. EP divides
expert memory by the EP world size; it does not make MoE memory-free. The
model's total capacity (64× FFN parameters) is preserved; it's just distributed.

The tradeoff is communication: four all-to-all operations per MoE layer per
training step (dispatch + return, forward + backward). On NVLink, these are
fast; cross-node on InfiniBand, they become the dominant cost at large EP
world sizes.

---

## EP + Other Parallelisms

**EP + TP**: these aren't really independent axes. EP needs a group of GPUs to
spread its experts across, and in practice that group is *carved from a parallel
axis you're already using* rather than a brand-new set of GPUs — frameworks differ
on which one (DeepSpeed splits it out of the data-parallel group; Megatron's
"parallel folding" repartitions the GPUs the attention layers use for TP and CP).
Whether each expert is *also* TP-sharded is a separate knob. Exactly how the group
is laid out is a device-mesh design decision; the deep dives go deeper.

**EP + ZeRO**: ZeRO shards expert parameters within the expert-replica group
(the ranks that hold the same expert) rather than the full DP group — otherwise
it would try to shard an expert across ranks that don't even have it. The two
have to agree on who owns the expert-gradient reduction so it happens exactly
once.

**EP + PP**: expert parameters belong to the PP stage that contains the MoE
layer. The EP all-to-all dispatch happens entirely within that stage — it doesn't
cross stage boundaries.

---

## What's Next: Putting It All Together

MoE and expert parallelism is the last individual technique in this series — but
real frontier runs never pick just one. They stack DP, ZeRO, TP, SP, PP, CP, and
EP together, each solving a different dimension of the memory and compute problem.

**[Part 10 — Putting It All Together](putting-it-all-together.html)** steps back
across all nine techniques: which ones shard the *model* and which shard the
*activations*, how each layer actually gets split, what each costs in memory and
communication, and which to reach for at a given scale. It closes with two real
open models — Qwen3 and DeepSeek-V3 — and the very different parallelism recipes
they chose to train at scale.
