# Implementation plans

One file per plan, `NNN-slug.md`, numbered in the order they were started.

A plan is **what we are going to do, in order**. It is working material: it gets checked
off as slices land, and it is deleted once the work is done and the code speaks for
itself. Do not curate plans — a stale plan is worse than no plan.

Decisions do not live here. The moment a plan contains a choice with lasting
consequence ("hard-modelling, not soft MCR"), that choice becomes an ADR in
[`../docs/decisions/`](../docs/decisions) and the plan links to it. The plan dies; the ADR
does not.

Structure that has earned its place:

```markdown
# NNN — <what this builds>

## Goal
One paragraph. What is true when this is finished that is not true now.

## Non-goals
What this deliberately does not do. Usually the more useful list.

## Slices
Each slice is independently landable and leaves the repo green.

- [ ] 1. ...
- [ ] 2. ...

## Open questions
Things that block a slice, and who or what resolves them.
```
