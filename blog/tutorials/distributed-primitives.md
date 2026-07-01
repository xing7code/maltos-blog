---
layout: post
title: "Distributed Primitives for LLM Training"
description: "Before DDP, ZeRO, or tensor parallelism, you need the basic language of distributed training: ranks, world size, process groups, and the collective operations that move tensors between GPUs."
category: Pretraining Concepts · Part 3 of 9
date: 2026-06-11
read_time: 11 min read
---

# Distributed Primitives for LLM Training

Before you can understand DDP, ZeRO, tensor parallelism, or expert parallelism,
you need a small vocabulary for how GPUs cooperate. Most distributed training
systems are built from the same few ideas: a set of ranks, one or more process
groups, and a handful of collective communication operations.

This article introduces those primitives from first principles. The goal is not
to memorize API names. It is to build the mental model you will reuse in every
later article in this series.

---

## Ranks, World Size, and Process Groups

In distributed training, each GPU usually runs its own process.

- **Rank**: the integer id of this process within some distributed job or group
- **World size**: the total number of participating processes
- **Process group**: a subset of ranks that communicate with one another

For a 4-GPU single-node DDP run, you typically have:

```text
world_size = 4
ranks = {0, 1, 2, 3}
dp_group = {0, 1, 2, 3}
```

In more complex jobs, the same rank can belong to multiple groups at once. For
example, in a `dp=2, tp=2` job on 4 GPUs:

```text
global ranks: 0 1 2 3

dp groups:
  {0, 2}
  {1, 3}

tp groups:
  {0, 1}
  {2, 3}
```

The key idea is that collectives do not run "over the whole cluster" by
default. They run inside a specific process group.

---

## Collective Communication

A **collective** is an operation that involves every rank in a group. Instead
of rank 0 manually sending tensors to rank 1, rank 2, and rank 3, all ranks
call the same collective and the backend performs the communication pattern.

The most important collectives for LLM training are:

| Collective | What every rank starts with | What every rank ends with | Typical use |
|---|---|---|---|
| `broadcast` | one rank has the source tensor | all ranks have the same tensor | parameter init, config sync |
| `all_reduce` | each rank has a tensor | each rank gets the reduced result | DDP gradient averaging |
| `all_gather` | each rank has one shard | each rank gets the full concatenated tensor | TP / ZeRO parameter assembly |
| `reduce_scatter` | each rank has a full tensor contribution | each rank gets one reduced shard | ZeRO / sharded gradient reduction |
| `all_to_all` | each rank has pieces for every other rank | each rank receives the pieces addressed to it | MoE token dispatch |

Two operations dominate the rest of this series: `all_reduce` and the
`reduce_scatter + all_gather` pair.

---

## Point-to-Point Communication

Not every distributed operation is a collective.

A **point-to-point (P2P)** operation sends data from one rank directly to
another specific rank. The receiving rank posts the matching receive. In
PyTorch distributed APIs, this usually appears as `send` / `recv` or the
non-blocking `isend` / `irecv` pair.

```text
rank 0 --send tensor--> rank 1
```

This matters whenever communication is structured as a chain, ring, or
stage-to-stage handoff rather than "everyone participates in one group
operation."

The most important examples in this series are:

- **Pipeline parallelism**: stage `i` sends activations to stage `i+1`, and the
  downstream stage sends activation gradients back during backward
- **Ring attention / context parallelism**: each rank sends its current KV block
  to one neighbor and receives the next block from the other neighbor
- **Runtime overlap patterns**: non-blocking `isend` / `irecv` let communication
  be enqueued earlier, with `wait()` deferred until the tensor is actually needed

So the real primitive split is:

- **Collectives**: one operation, all ranks in a process group participate
- **P2P**: one sender and one receiver coordinate directly

Both matter. DDP and ZeRO are mostly collective-heavy; pipeline parallelism and
ring attention rely heavily on P2P.

---

## Broadcast

`broadcast` copies one tensor from a source rank to every other rank in the
group.

```text
before:
  rank 0: [1, 2, 3]
  rank 1: [?, ?, ?]
  rank 2: [?, ?, ?]

after broadcast(src=0):
  rank 0: [1, 2, 3]
  rank 1: [1, 2, 3]
  rank 2: [1, 2, 3]
```

This is conceptually the simplest collective. It shows up when:

- rank 0 loads a checkpoint and needs other ranks to receive metadata
- a root rank initializes a tensor and all others need the same value
- pipeline stages or expert groups need shared configuration

---

## All-Reduce

`all_reduce` is the workhorse of plain data parallelism.

Each rank starts with its own tensor. The backend reduces them elementwise
(usually by sum or average) and writes the same result back to every rank.

```text
before:
  rank 0: [1, 2]
  rank 1: [3, 4]
  rank 2: [5, 6]

after all_reduce(sum):
  every rank: [9, 12]
```

For DDP gradient averaging, each rank computes gradients on a different batch,
then all-reduces the gradients before `optimizer.step()`.

```python
loss.backward()
dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
param.grad /= dist.get_world_size(dp_group)
```

Conceptually, `all_reduce` says:

1. everyone contributes
2. the contributions are combined
3. everyone receives the same final tensor

That is exactly what replicated data parallelism needs.

---

## All-Gather

`all_gather` does the opposite kind of job. Instead of combining values, it
assembles shards.

```text
before:
  rank 0: [a0, a1]
  rank 1: [a2, a3]
  rank 2: [a4, a5]

after all_gather:
  every rank: [a0, a1, a2, a3, a4, a5]
```

This matters whenever each rank owns only a piece of a larger logical tensor:

- tensor parallelism gathers partial outputs
- ZeRO-3 gathers parameter shards before a module runs
- context or expert parallel variants gather data that was partitioned earlier

`all_gather` increases local memory because every rank ends up with the full
tensor.

---

## Reduce-Scatter

`reduce_scatter` combines reduction and sharding in one step.

Each rank contributes a full tensor, the backend reduces them elementwise, and
then scatters the reduced result so each rank keeps only its assigned slice.

```text
before:
  rank 0: [1, 2, 3, 4]
  rank 1: [5, 6, 7, 8]

reduced sum:
  [6, 8, 10, 12]

after reduce_scatter:
  rank 0: [6, 8]
  rank 1: [10, 12]
```

This is why ZeRO prefers `reduce_scatter` over `all_reduce`: after gradient
reduction, each rank only needs the shard it is responsible for updating.

Compared with `all_reduce`, `reduce_scatter` trades "every rank gets the whole
answer" for "every rank gets only its shard of the answer."

---

## Why `All-Reduce = Reduce-Scatter + All-Gather`

A useful identity is:

```text
all_reduce = reduce_scatter + all_gather
```

Not as Python syntax, but as a communication pattern.

If you:

1. reduce a tensor and scatter shards across ranks
2. then all-gather those reduced shards back

every rank ends up with the same result an `all_reduce` would have produced.

This identity matters because many optimized systems restructure communication
around sharded states:

- DDP conceptually wants `all_reduce`
- ZeRO often wants only the `reduce_scatter` half
- TP and ZeRO-3 often need the `all_gather` half later, but only when a full
  logical tensor is temporarily required

So the same communication building blocks reappear under different parallelism
strategies.

---

## Choosing the Right Primitive

A good shortcut is to ask two questions:

1. Do all ranks need the full result, or only a shard?
2. Are we combining values, or assembling shards?

That gives a quick decision table:

- Combine values, full result on all ranks: `all_reduce`
- Combine values, sharded result: `reduce_scatter`
- Assemble shards into a full tensor: `all_gather`
- Copy one tensor from one rank to all ranks: `broadcast`
- Exchange per-destination pieces: `all_to_all`
- Send one tensor from rank A to rank B: `send` / `recv` or `isend` / `irecv`

This is the real "distributed primitives" mental model. Different parallelism
strategies mostly differ in when they invoke these operations and on which
tensors.

---

## Backend and Transport

The API names stay the same across hardware, but the backend matters.

- **NCCL**: standard backend for GPU training; uses NVLink, PCIe, InfiniBand, or RoCE underneath
- **Gloo**: often used for CPU testing and debugging

On a single machine, collectives usually run over NVLink or PCIe. On multiple
machines, they span NICs and switches, so communication cost becomes much more
visible. That is why large-scale training systems care so much about overlap,
bucket sizing, and sharding.

---

## What Comes Next

The next article uses these primitives to explain DDP. In plain replicated data
parallelism, the critical operation is gradient `all_reduce`: every rank computes
its own gradients, then all ranks average them before the optimizer step.

Later articles reuse the same vocabulary in different ways:

- TP uses collectives to combine partial matmul results
- ZeRO replaces replicated gradient reduction with `reduce_scatter`
- ZeRO-3 and TP use `all_gather` to materialize full tensors only when needed
- PP uses P2P `isend` / `irecv` to move activations and gradients between stages
- Ring attention uses P2P neighbor exchange instead of all-gathering the full KV cache
- MoE uses `all_to_all` to route tokens to experts
