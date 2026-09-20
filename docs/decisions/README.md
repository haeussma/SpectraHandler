# Architecture decision records

One file per decision, `NNNN-slug.md`. Written by the `/spec-link` command, which also
wires the matching `~/brain/wiki/methods/` page in the other direction.

An ADR is **why we chose something, kept forever** — the counterpart to a plan in
[`../../plans/`](../../plans), which records what we are doing and is deleted when done.
Write one whenever a choice would be expensive to reverse, or whenever someone six months
from now would otherwise reopen it: the inference approach, the scope boundary against
`mcrals`, a deviation from what a paper actually describes.

These files are **not published**. Astro only builds `docs/src/content/docs/`, so this
directory is versioned with the code and invisible to the docs site. If a decision is also
worth explaining to users, that is a separate page under `src/content/docs/`.

Required shape, per `/spec-link`:

```markdown
---
status: proposed | accepted | superseded
date: YYYY-MM-DD
source_paper: doi:10.xxxx/...    # or omit
brain_page: ~/brain/wiki/methods/<slug>.md
implements: [src/spectrahandler/<module>.py]
---

# NNNN — <decision>

## Context
What the science says, citing the paper and the brain page.

## Decision
What we actually coded, and where it deviates from the source.

## Consequences
Assumptions this bakes in, and the limits of its validity.
```
