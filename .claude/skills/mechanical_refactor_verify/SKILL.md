---
name: mechanical_refactor_verify
description: Make a mechanical refactor (file split, function move, module extraction, rename) checkable instead of eyeballed, and prove it changed no behavior. Separate pure relocations from semantic edits, reconstruct the relocation so a reviewer can diff it to nothing, and gate numerics with a bit-identical loss comparison. Use when moving/splitting/renaming code or reviewing such a PR.
---

# Verifying a mechanical refactor

A mechanical refactor (move, split, extract, rename) must change **no behavior**.
The repo's standard for that is bitwise-identical loss and grad_norm
(`AGENTS.md`, "Validating Numerics"). This skill makes the "mechanical" claim
auditable so the numerics check is backed by structure, not hope.

## Separate the two kinds of change

Split the work so each commit is one kind:

- **Pure relocation** -- bytes move verbatim: a function moves file to file, a
  block is extracted into a helper called in the same place, a symbol is renamed
  everywhere. A reviewer should be able to reconstruct it from a rule and diff
  the result to zero.
- **Semantic edit** -- anything that changes what runs: a new default, a reorder,
  a merged branch, a signature change. These need their own justification and
  numerics proof.

Never mix them in one commit. A rename buried in a logic change is the classic
way a "no-op refactor" silently alters behavior.

## Make the relocation reconstructable

For each pure relocation, write down the transformation precisely enough to
replay it:

- **Move/extract**: the exact source range, the destination, and the import /
  call edit that re-wires it. Reconstruct by applying only those, then
  `git diff` the reconstruction against the PR -- it must be empty.
- **Rename**: the old and new name and the scope. Reconstruct with a
  whole-word, scope-aware replace; diff to zero. Watch for partial-word hits and
  strings.
- Keep moved code byte-identical: no reflow, no comment edits, no reordering in
  the same commit. If formatting must change, that is a separate follow-up commit.

Audit all callsites (`AGENTS.md` principle #6): a move or rename of shared code
in `torchtitan/distributed/` or `torchtitan/models/common/` must update every
model variant (llama3, llama4, qwen3, deepseek_v3, flux, ...).

## Gate numerics bit-identical

A pure relocation must produce identical loss and grad_norm before vs. after:

```bash
scripts/loss_compare.py <before-ref> <after-ref> --assert-equal
# runs with --debug.deterministic; compares from TensorBoard, since stdout
# prints only 5 significant digits
```

Run it across the parallelism configs the moved code touches (distributed code
is easy to break silently: `.claude/rules/distributed.md`). If the losses are
not identical, the refactor was not mechanical -- find the accidental semantic
change.

## Deliver

Small, reviewable commits: relocations first (each diffable to zero), semantic
edits after (each with its numerics proof). State in the PR which commits are
pure relocations and how to reconstruct them.
