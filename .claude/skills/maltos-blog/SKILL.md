---
name: maltos-blog
description: Authoring, polishing, and maintaining the MALTOS technical blog & manual (the maltos-blog repo — Tutorial, Deep Dive, and Learn MALTOS tracks documenting the MALTOS LLM-training framework). Use when writing or editing articles/manual pages, running the multi-round reviewer-persona polish workflow, verifying claims against the ../maltos source, building/previewing the site, or working on its layout, theme, and diagrams.
---

# Working on maltos-blog

`maltos-blog` is the technical writing companion to **MALTOS**, a modular,
composable runtime for LLM pretraining (TP/SP, DP/DDP, PP, CP, EP, ZeRO 1/2/3,
sharded checkpointing, phase-based plugin runtime). The blog explains the
system and serves as its manual. The framework source lives in a sibling repo
at `../maltos` (relative to this repo root).

## The #1 rule: verify every claim against ../maltos source

This is the single most important habit and the one that has caught the most
errors. **Before writing any technical assertion** — an API signature, a field
name, a validation rule, control flow, a default, a number — grep/read the
actual file in `../maltos`. Do not write from memory or plausibility.

- The codebase has drifted from prose more than once (e.g. `RuntimePhase.TRANSFORM_MODEL`
  is in the enum but never fired; `GradClipPlugin` computes a *local* norm, not
  a global one, so it's wrong under sharding; the CLI gained `--ep-size` after a
  doc said EP wasn't exposed).
- When you find the code contradicts a nice story, **say so honestly in the
  article** — documenting a real limitation/bug is more valuable than a clean
  fiction. The deep dives explicitly call out gaps ("the honest gap…", "one
  honest caveat…").
- Each Learn page opens with a `> **Source(s)**: …` line naming the exact files
  it documents. Keep that accurate; it's the page's verifiability contract.

Key source areas in `../maltos`:
`runtime/core.py` (RuntimeCore: setup/run_step/step_optimizer, phases),
`runtime/plugin.py` (RuntimePlugin base, PluginId, spec protocols),
`runtime/mesh.py` (MeshConfig, process groups, derived DCP/EREP axes, EP reuse),
`runtime/plugins/*` (ddp, tp, sp, pp, cp*, ep, zero1/2/3, zero_common, precision,
grad_clip, perf_metrics), `parallel/*` (ParallelPlan, specs, pipeline, context,
expert, schedule), `train/trainer.py` (Trainer, TrainerConfig, retention),
`state/*` (StateManager, checkpoint manifest/atomic write), `data/*`,
`models/*` (tiny_transformer, llama, tiny_moe_transformer — reference models),
`tools/pretrain.py` + `tools/prepare_token_shards.py`, `tests/*`, `docs/*`,
`profiles/*` (perf snapshots + profiler trace bundles).

## The polish workflow (how we iterate)

Articles are taken through **3–5 rounds of iterative review**, each round:
verify against source → apply edits directly to the files (don't just suggest) →
simulate reader personas → apply their feedback. Match the reviewer persona to
the track:

- **Tutorials** → junior RS / SWE who know ML & PyTorch but are new to AI infra.
  Watch for: undefined jargon, missing prerequisites, broken reasoning chains,
  "but why?" gaps, code with unexplained variables.
- **Deep dives** → frontier-lab senior/staff infra engineers. Be adversarial:
  challenge the thesis, demand derivations (e.g. gradient correction factors),
  prior-art positioning (DeepSpeed/Megatron/DTensor), and honesty about what
  doesn't compose / isn't fast yet.
- **Learn · User Guide** → junior RS *users* trying to run their own model.
  Watch for first-hour traps (silent logger default, torchrun vs python launch,
  data must exist first, expected-output cues).
- **Learn · Internals** → senior/staff contributors. Code-accurate; verify phase
  firing, plugin ordering, mesh math, test tolerances against source.

Calibrate depth: tutorials teach concepts ("don't go too deep"); deep dives go
deep on design rationale; the manual is task-oriented how-to/reference and
**must not re-teach concepts** — it links to the tutorial/deep-dive instead.

After each round, give a short report: what was found, what changed, what
improved. Keep a running TODO of rounds.

## Repo geography

```
maltos-blog/
├── index.html                 hand-maintained homepage (cards per track)
├── blog/index.html            article index (hand-maintained)
├── blog/tutorials/*.md        Tutorial series (8 parts) + index.html
├── blog/internals/*.md        Deep Dive series (8 articles) + index.html
├── blog/learn/*.md            Learn MALTOS manual (12 pages) + index.html
│     User Guide: quickstart, bring-your-own-model, configuring-parallelism,
│                 data-and-token-shards, training-checkpoints-and-resume,
│                 errors-and-troubleshooting
│     Internals:  repo-map, runtime-core-walkthrough, writing-a-plugin,
│                 mesh-and-process-groups, test-architecture,
│                 adding-tests-and-profiling
├── _layouts/{default,post,manual}.html   Jekyll layouts
├── _includes/{manual_nav,theme_script}.html
├── styles.css                 single stylesheet; CSS-variable themed
├── assets/, blog/assets/      SVG diagrams + W&B PNGs (both dirs; keep in sync)
└── local_notes/               gitignored working notes + the preview tooling
      preview_site.py          the local preview build/server
      .preview-venv/           its venv (markdown, pyyaml, pygments)
      perf_roadmap_8x4090.md   perf findings + recommended reruns (carry-over)
      deep_dive_release_plan.md, review_8x4090_profile_post.md
```

Article frontmatter: `layout` (`post` or `manual`), `title`, `description`,
`category` (e.g. `Tutorial · Part 4 of 8`, `Deep Dive`,
`MALTOS Guide · User Guide`, `MALTOS Guide · Internals`), `date`, `read_time`.
The first markdown `# H1` repeats the title and is stripped by the renderer.
Cross-link articles liberally with relative `.html` links.

## Build & preview (TWO render paths — keep both working)

The site has two independent renderers; **layout/theme changes must be made in
both** or preview and production diverge:

1. **Jekyll** (production): `_layouts/`, `_includes/`, kramdown + Rouge
   highlighting, `baseurl: /maltos-blog`, uses `relative_url`.
2. **Local preview**: `local_notes/preview_site.py` — Python markdown + Pygments,
   absolute `/` paths. Build dispatches `.md` by `layout`: `manual` →
   `_render_manual`, else `_render_post` (and `_render_post` itself renders
   internals/tutorials with the manual sidebar shell). Static `index.html`
   pages are copied as-is.

Run the preview with its venv (NOT system python — it lacks markdown/pygments):
```bash
local_notes/.preview-venv/bin/python local_notes/preview_site.py --build-only   # rebuild
local_notes/.preview-venv/bin/python local_notes/preview_site.py --port 4001    # build + serve
```
A server is usually already running on a 400x port; after editing `.md`/CSS,
rebuild with `--build-only` and refresh (browser CSS caching has caused
"changes didn't apply" confusion — hard-refresh). Verify served output with
`curl -s http://127.0.0.1:<port>/...` since you can't see rendered pixels.

## Layout, theme, highlighting facts

- **Theming**: CSS variables in `:root` (light) and `html[data-theme="dark"]`
  (dark). A warm light palette + dark mode both exist. Never hardcode colors that
  must adapt — route them through vars (`--bg`, `--panel`, `--ink`, `--body-text`,
  `--muted`, `--line`, `--accent*`, `--code`, `--code-bg`, `--header-bg/line`,
  `--shadow`). Dark-incompatible hardcoded colors were a recurring bug.
- **Theme toggle** lives in `_includes/theme_script.html` (single source): applies
  saved theme before paint, injects a ☾/☀ button into `.nav-links`, persists to
  `localStorage`. It's included by the Jekyll layout, both preview renderers, and
  injected into the static index pages. If you add a new top-level HTML page, it
  needs this script in `<head>`.
- **Manual/sidebar layout**: a centered grid (`.manual-shell` = sticky TOC +
  `.manual-content`), TOC stays while scrolling, collapsible via the
  `.manual-toc-toggle` (desktop: collapse & reclaim width; <1000px: overlay
  drawer). The TOC list is `_includes/manual_nav.html`.
- **Syntax highlighting** requires Pygments in the preview venv; token colors are
  defined in `styles.css` for both `.codehilite` (preview/Pygments) and
  `.highlight` (Jekyll/Rouge). Tuned for the dark code background.
- **Diagrams** are hand-authored SVG in the blog's palette (teal `--accent`
  family on light panels). Referenced via `<div class="article-figure"><img></div>`.
  Keep copies in BOTH `assets/` and `blog/assets/` (article pages reference
  `../assets/` or `/blog/assets/` depending on path).

## MALTOS domain cheat-sheet (mental model; still verify before writing)

- **Runtime**: `Trainer` (policy: when to log/checkpoint/step) drives
  `RuntimeCore` (execution: phases, plugin coordination, optimizer step). Trainer
  stays tiny; all distributed behavior is in plugins.
- **Phases** (fired in topologically-sorted plugin order): SETUP, TRANSFORM_MODEL
  (enum exists but unused — transforms run directly in setup), PRE_MICROBATCH,
  PRE/POST_FORWARD, PRE/POST_BACKWARD, PRE/POST_STEP, PRE_SAVE, POST_LOAD.
- **Plugins** declare `requires`/`runs_after`/`runs_before` (graphlib topo-sort)
  and optional `owns_optimizer`. Extension is via plugins + module hooks +
  `register_post_grad_reduction_callback` (CP/EP/SP attach gradient semantics
  without patching reducers).
- **Optimizer**: factory-only (`optimizer_factory`, never a prebuilt optimizer),
  built AFTER model transforms; exactly one owner (runtime, or one ZeRO plugin);
  the step itself always executes in one shared `step_optimizer()` path —
  "ownership" decides *which* optimizer, not *who calls* `.step()`.
- **Mesh**: axes DP/TP/PP/CP/EP plus derived **DCP** (DP×CP, non-expert ZeRO
  sharding) and **EREP** (expert-replica group). Gradient **correction factors**
  reconcile `ReduceOp.AVG` over a group with the desired divisor (e.g. CP slices
  sum, DP replicas average); getting these wrong silently scales gradients —
  which is why equivalence tests compare gradients, not just loss.
- **EP** = a parameter-role problem: `ParamRole.EXPERT` vs `SHARED`; experts are
  EP-sharded (never TP-sharded), reduce over EREP; shared params reduce over
  DP/DCP. Four EP reuse layouts (reuse_tp/reuse_cp).
- **Checkpoints**: per-rank sharded files + a `manifest.json` (world_size,
  per-rank entries, `optimizer_source_ranks`, atomic `.tmp`→rename). Resume
  validates topology; no resharding on topology change yet. Exact resume includes
  dataloader cursor + RNG + plugin state.
- **Testing**: three layers — smoke / single-feature equivalence / full-stack
  matrix; gloo (CPU, default) before NCCL; equivalence = match an unsharded
  baseline on loss/grads/post-step params within stated tolerances
  (full-stack 1e-3 loss, 1e-5 grad/param; resume 1e-6; midstep 5e-5).
  `tests/run_single_feature.sh`, `tests/run_matrix.sh`, `tests/profile_train_perf.py`.

## Performance status (as of the 8x4090 milestone)

The system is correct (NCCL matrix passes) but slow on the PCIe 8x4090 box.
Trace analysis (see `local_notes/perf_roadmap_8x4090.md`): ZeRO-3 is
communication-bound (NCCL ~95% of GPU kernel time, ~1% MFU); root causes are
per-`nn.Linear`/per-RMSNorm bucket granularity (654 tiny all-gathers/step),
fp32 reduce-scatter, and per-microbatch re-gather under PP. Top fixes, in
leverage order: coarsen `wrap_cls` to decoder-layer granularity, bf16 gradient
reduce-scatter, keep params materialized across microbatches, and "don't shard
what fits" (a 1.3B model doesn't need ZeRO-3). Suggested reruns are listed in
that note; the profile article (`what-actually-gets-slow-on-8x4090-llm-training.md`)
has a "missing experiment" hook for the CP long-seq sweep.

## Tone

Precise, technical, honest. No marketing fluff. Numbers must be defensible
(derive or cite). Prefer "here is the real tradeoff / limitation" over polish.
Apply edits directly to files; report what changed.
