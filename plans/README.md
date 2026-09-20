# Implementation plans

One **directory** per plan, `NNN-slug/`, numbered in the order they were started.

```
plans/001-bayesian-curve-resolution/
  spec.md       # what we are building and why it is shaped this way
  plan.md       # the executable task breakdown, derived from the spec
  research.md   # prior art and background, when there is any
```

A plan is **what we are going to do, in order**. It is working material: it gets checked
off as tasks land, and the directory is deleted once the work is done and the code speaks
for itself. Do not curate plans — a stale plan is worse than no plan.

## spec.md and plan.md are different documents

The split is not bureaucracy; the two are read by different people at different times.

**`spec.md`** argues. It states scope and non-goals, the data contract, the model, the
complexity ladder, and the open questions. A human reads it to decide whether the
approach is right. It is written once and amended as evidence arrives.

**`plan.md`** executes. Numbered tasks, each with files, interfaces, a failing test, the
implementation, the command to run, and a commit. It follows the `superpowers:writing-plans`
format and is executed task-by-task by `superpowers:subagent-driven-development` or
`superpowers:executing-plans`. An agent reads it with no other context, so it contains
real code — never "TBD", never "add error handling", never "similar to Task 2".

Write `plan.md` only as far as the next real gate. A plan for work whose shape depends on
the outcome of earlier work is fiction.

## Decisions do not live here

The moment a plan contains a choice with lasting consequence — "dimension order is
(run, time, wavelength)" — that choice becomes an ADR in
[`../docs/decisions/`](../docs/decisions) and both documents link to it. The plan dies;
the ADR does not.

An ADR that is still `status: proposed` and that a task depends on is a **blocking
precondition**, and `plan.md` says so at the top. Do not start the task.
