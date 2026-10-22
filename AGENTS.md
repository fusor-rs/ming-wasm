# AGENTS.md

Rules for coding agents in this repository. They are not suggestions: a change
that breaks one is not finished, however well it works. When a rule and the task
truly conflict, stop and say so instead of quietly bending either.

ming-wasm is the Ming-3.0-tts-0.5B text-to-speech app: the model running in the
browser on the CPU, sealed from the network. The README describes what it does;
don't invent scope, features or options beyond the task.

## Repository

- `vendor/fusor-ai/models/ming/runtime`: the wasm-bindgen surface of the model
  (bundle parts in, PCM out), built as four browser runtimes.
- `vendor/fusor-ai/models/ming/cli`: the native CLI `ming` (`compile`, `say`,
  `bench`, `parity`).
- `app/`: the page, a [fusor](https://github.com/fusor-rs/fusor) app with its own
  Cargo workspace (fusor pins wasm-bindgen 0.2.117, the runtime 0.2.129): the
  form and progress as Rust signals in `src/app.rs`, the template in
  `web/index.html`, the engine bridge and player in `web/*.js`, and the engine
  worker in `engine/worker.js`, beside the runtimes `just runtimes` builds.
- `scripts/`: the checkpoint download, the golden dump, the Node smoke test and
  the ASR check; `e2e/run.mjs`: the headless Chrome run.
- `benchmarks.md` and `benchmarks/`: the measured speed against the PyTorch
  reference and other runtimes, with the harness that reproduces it.
- [fusor-ai](https://github.com/fusor-rs/fusor-ai): the model (`ming`), its
  compiler (`ming-compile`), the kernels, blocks and bundle format come from
  `vendor/fusor-ai`, a git submodule pinned to a commit pushed to fusor-ai, with
  the browser loader and compute pool, the runtime build, the import check and
  the dev server.

What belongs where: model code (dimensions, tensor names, wiring, sampling,
prompts, the text frontend) and anything another model could reuse go to
fusor-ai, following its `AGENTS.md`. This repository holds what turns the model
into an app: the runtime surface, the CLI, the page, the checks against the
reference and the releases. A change to fusor-ai lands there first; this
repository then moves its submodule to it.

Constraints that hold everywhere:

- Rust 1.99 (edition 2024) is the minimum, as in fusor-ai. Threaded runtimes
  build with `nightly-2025-11-15` and build-std; fusor-ai's `build-runtime.sh`
  passes `--ignore-rust-version` for them, so the wasm code path must keep
  building on that nightly. Don't move the pin without saying why in the change.
- `unsafe` is denied workspace-wide, with no exception: the getrandom backend that
  needed it is fusor-ai's (its `runtime` crate).
- The runtime imports nothing that reaches the network, storage, a clock or
  entropy. fusor-ai's `scripts/check-wasm-imports.mjs` is the check (`just
  runtimes` runs it on every build); never widen its allowlists to make a build
  pass.
- fusor-ai comes only from `vendor/fusor-ai`: path dependencies in
  `[workspace.dependencies]`, never a crates.io version, a `[patch]` table or a
  sibling checkout. The submodule moves to another pushed fusor-ai commit in a
  change of its own, with the parity numbers and the e2e result it gave.
- Every crate shares the version in `[workspace.package]`.

## The standard

Write the smallest correct change that a careful senior reviewer would merge
without comments. Agent-written code is measurably more verbose, more duplicated
and more eroded than human code, and it gets worse with every iteration. Assume
your first draft has those faults and remove them before you finish.

- **Less code, never compressed code.** Between two correct, equally readable
  solutions, take the one with less logic. Deleting code is progress. Report
  the net line change of your diff; every added line must be needed by the
  task. Squeezing the same logic into fewer, denser lines is not less code: it
  is worse code. When brevity and readability conflict, readability wins.
- **Write for the maintainer, not the grader.** Passing the checks is the
  minimum, not the goal. The code must still be easy to read, extend and debug
  six months from now by someone who never saw this task.
- **Read before writing.** Before adding a function, type, constant or
  dependency, search the repository for one that already does it
  (`rg -n '<name or concept>'`) and use or extend it.
- **Verify, don't assume.** Check every API you call against its source or
  docs.rs (`~/.cargo/registry/src/*/<crate>-<version>/`). Never call a method or
  add a crate you have not confirmed exists. Before calling code unused,
  `rg` for its uses.
- **Run it.** "It compiles" is not evidence it works. Execute the behavior you
  changed, or a test that fails without your change, and say what you ran.
- **Fix causes, not symptoms.** Reproduce a bug first. Don't add a check,
  retry, sleep or fallback to make a failure disappear without knowing why it
  happened.

## Scope and integrity

The most common complaint about strong coding models is that they do too much:
they rewrite code nobody asked about, break things next to the change, and hand
back half-finished work or work that passes by cheating. None of that is
allowed.

- **Do exactly the task.** Change the smallest region that achieves it. Don't
  rewrite a working function, file or module to make a small change, and don't
  refactor, rename, reformat or "modernize" code the task doesn't need.
- **Don't break what's next to it.** Before editing anything shared (a
  function with other callers, a type, a config value, a stylesheet rule, a
  query), find every other user with `rg` and confirm they still behave the
  same. Then run the full checks, not only the tests near your change.
- **Finish or say so.** Never present partial, stubbed or untested work as
  done. If you can't complete the task, stop and state exactly what is missing.
- **Don't expand the task.** Don't start adjacent work, and don't end with
  offers of more ("Should I also…?"). Problems you noticed outside the task go
  in one line of your summary, unfixed.
- **Never game a check.** Don't special-case test inputs, hardcode expected
  outputs, read tests to reverse-engineer answers, or delete, skip, `#[ignore]`,
  loosen or rewrite a test, assertion or fixture to make it pass. If a test
  looks wrong, stop and say why instead of changing it.
- **Never fake the result.** Build the mechanism the task asks for, not
  something that only looks like its output: no canned or hardcoded data in
  place of real computation, no stand-in or mock in a production path, no
  output staged to resemble success. If the real thing can't be built, say so.
- **Never destroy what you didn't create.** No deleting files or data outside
  the task, and no `rm -rf`, `git reset --hard`, `git clean`, `git checkout --`
  on uncommitted work or force pushes without explicit permission.
- **Match the repository.** Follow the structure, naming and idioms already
  here. Don't introduce a second style, layout or architecture for the same
  kind of thing.

## Hard rules

### Duplication and abstraction

- No near-duplicate blocks. Code that repeats with small edits becomes one
  function, loop or table. Two copies of the same concept is already one too
  many; the same shape doing different concepts may stay apart.
- One concept, one implementation. Never add a second path beside an existing
  one (a second parser, validator, formatter or error type); extend the first.
- One source of truth per constant. A limit, name, path or version is defined
  once and referenced everywhere else, including in messages and tests.
- No speculative generality. No trait with one implementation, generic with one
  instantiation, builder for fewer than four optional fields, config option,
  flag or parameter without a caller that needs it today.
- No trivial wrappers. A function that only forwards to another, or renames it,
  is inlined. A new function must remove duplication or name a real concept.
- No `utils`, `helpers`, `common` or `misc` modules, and no `Manager`, `Helper`,
  `Util`, `Data`, `Info`, `Impl` or `Wrapper` type names. Name things after the
  domain concept they hold.

### Readability

Code is read far more than it is written. It must read top to bottom without
decoding.

- One statement per line, formatted by `rustfmt` with default settings. No
  chained one-liners that hide several steps, and nothing that looks minified.
- Names are full words from the domain: `pending_job`, not `pj`, `tmp`, `res`
  or `data2`. Single letters only for loop counters and coordinates.
- No magic numbers or strings in logic. Give them a named constant.
- Long text never sits in one-line literals. Templates, CSS and other markup
  live in their own file (`include_str!`) or in a multi-line literal laid out
  the way the language is normally written.
- Propagate errors with `?`. Never repeat check-and-unwrap boilerplate on every
  call.
- Check a repeated guard once, at the loop or function head, not before every
  step.
- When branches build nearly the same value, build it once and vary only the
  part that differs.
- Many similar items (materials, routes, columns, cases) are data: a table and
  one loop, never dozens of near-identical statements.
- Decompose by concept. A function reads as one level of abstraction; an
  orchestrating function is a short sequence of named steps. A file holds one
  concept; never pile unrelated concepts into one giant file or function.

### Functions and types

- Functions stay under 60 lines, with at most 5 parameters and at most 3
  levels of nested blocks inside a method (`excessive_nesting` counts the
  enclosing `impl` or `mod` as one more). Split by concept, not by line count;
  flatten with early returns and `let … else` before extracting functions.
  Values that travel together (`x, y, width, height`) or several parameters of
  the same type become a struct.
- No boolean parameters that switch behavior (`render(node, true)`). Use two
  functions or an enum.
- Make invalid states unrepresentable: enums instead of string or integer tags,
  newtypes for identifiers and units, no `Option<Option<T>>`, no paired
  `Option`s that must be both set or both empty.
- Default visibility is private, then `pub(crate)`. `pub` only for the intended
  public API.
- No single-use intermediate variables that only rename an expression, and no
  `let x = x.clone()` without a borrow that requires it.

### Errors

- Never swallow a failure. No `.ok()`, `unwrap_or_default()`, `unwrap_or(…)`,
  `let _ = …` or empty `match` arm on a `Result` unless absence is a real,
  expected outcome, stated in a one-line comment.
- No `unwrap()` outside tests. `expect("…")` only for invariants the code
  guarantees; the message states the invariant.
- Library code returns typed errors; `Box<dyn Error>` and `anyhow` belong only
  in binaries. Error messages say what failed and what to do about it.
- No defensive checks for impossible states. If the types already guarantee it,
  don't re-check it; if they don't, change the types.

### Comments: default zero

Agents over-comment more than anything else. The budget for comments is zero;
names and types carry the meaning. A comment is allowed only when deleting it
would let a competent reader make a wrong change:

- an invariant or ordering requirement the code relies on;
- a constraint that is not visible in the code (a protocol rule, a bug in a
  dependency), with a link when one exists;
- why an obvious alternative was not used.

Everything else is forbidden:

- Comments that say what the code does, restate a name or narrate steps
  (`// Step 1`, `// loop over rows`, `// return the result`).
- Section banners, commented-out code, and comments inside tests explaining
  each line.
- Doc comments that repeat the signature (`/// Creates a new Parser`,
  `/// Returns the name`). `///` goes only on public items, and only for what
  the signature can't say: units, errors, panics, invariants.
- Process comments about the change, the task, the agent or the old version
  ("now uses", "updated to", "fixed", "previously", "new:", "as requested").
  History belongs in the commit message.
- `TODO`, `FIXME`, `todo!()` or `unimplemented!()` in a finished change.

When an edit makes an existing comment wrong, fix or delete it in the same change.

Docs describe current behavior plainly. No work logs, progress files, AI
summaries, changelog tense ("now supports") or hedged non-claims ("not
guaranteed to", "may or may not").

### Dependencies and performance

- Prefer the standard library. Add a dependency only for real functionality the
  task needs, with an exact version, and confirm its API before use.
- No clone, `to_vec`, `collect` or `String` allocation added just to satisfy the
  borrow checker or to iterate once more. Borrow, or restructure.
- No `async`, `Arc`, `Mutex` or threads unless the task needs concurrency.
  Parallel work goes through `team` (one `Team::run` per component
  call, barriers inside), never a Rayon region per matmul.
- No allocation in a per-row or per-block kernel loop.
- No optimization without a measurement, and no pessimization either: don't
  rebuild, reparse or re-sort the same data in a loop.

### Tests: only what guards behavior

A test exists to fail when behavior breaks. Tests written for coverage, for
each function or for each branch are noise that hides the ones that matter.

- Write the fewest tests that would catch a real regression of the behavior you
  changed: usually one, table-driven when the behavior has variations. A bug
  fix gets exactly one test that reproduces the bug.
- Extend an existing test before adding a new one, and delete tests your change
  makes redundant. The model is checked stage by stage against the PyTorch
  goldens by `ming parity`; extend it rather than adding a parallel harness.
- Test observable behavior through the public interface. Never test private
  helpers, internal state or the order of internal calls.
- Don't test what cannot break independently: trivial constructors and
  getters, derived impls, the standard library, dependencies, the compiler, or
  anything the type system already guarantees.
- No tautological tests. Expected values are literals worked out independently,
  never computed by calling the code under test or by repeating its logic.
- Don't mock your own code. Fake only true external boundaries (network, clock,
  filesystem) when the real or in-memory one is impractical, and never assert
  on a fake you configured yourself.
- Every assertion must be able to fail. Assert exact values, not
  `result.is_ok()`, "not empty" or "doesn't panic". No assertions in callbacks
  that might not run.
- Every new test must fail without the change it covers. Check it: revert the
  change and run the test.
- No sleeps, wall-clock timing or test-order dependence.
- Test names state the behavior (`rejects_duplicate_keys`), not the function
  (`test_parse_2`).

### Suppressions

- Never silence the compiler or Clippy with `#[allow(…)]`. Fix the code. If an
  exception is truly right, use `#[expect(lint, reason = "…")]` with the reason.
- Never weaken a lint, threshold or check to make a change pass.

## Enforced lints

`[workspace.lints]` in the root `Cargo.toml` and `clippy.toml` beside it are the
lint configuration, the same as fusor-ai's; every crate opts in with
`[lints] workspace = true`. Clippy denies, among others, `unwrap_used`,
`allow_attributes`, `too_many_lines` (60), `cognitive_complexity` (15),
`excessive_nesting`, `min_ident_chars`, `wildcard_imports`,
`needless_pass_by_value`, `redundant_clone` and `undocumented_unsafe_blocks`.

## Checks

A change is done only when this passes:

```sh
just check
```

It runs formatting, the workspace tests, Clippy natively and for the runtime
in `wasm32-unknown-unknown` (SIMD128 and relaxed SIMD), docs, and syntax checks
of the scripts and the page. A change to the runtime, the page, the CLI or the
submodule also needs `just runtimes`, `just smoke`, `just parity` (the same
numbers as before unless the change is meant to move them) and `just e2e`.

## Final review

Long sessions make agents drift from these rules. Before reporting a task as
done, review your own diff, not your memory of it:

```sh
git diff --stat
git diff | grep -nE '^\+.*(unwrap\(\)|\.ok\(\)|unwrap_or_default\(\)|let _ =|#\[allow|todo!|unimplemented!|dbg!|TODO|FIXME)'
git diff | grep -niE '^\+\s*//.*\b(now|updated?|fixed|previously|new:|step [0-9]|as requested)\b'
git diff | grep -nE '^\+\s*(//|/\*)'
git diff | grep -nE '^\+\s*#\[test\]'
git diff | grep -nE '^\+.{101,}'
```

Every hit from the greps must be justified or removed: each added comment
against the comment rules, each added test against the test rules, each line
over 100 characters against the readability rules. Then re-read
each changed hunk against the hard rules above, and confirm:

1. Nothing duplicates code that already existed, inside or outside the diff.
2. Every new function, type, parameter and dependency has a current caller.
3. Every behavior change was executed or is covered by a test that fails
   without it, and no test covers a behavior another test already covers.
4. The diff contains nothing outside the task.
5. Each changed function reads top to bottom in one pass, with no dead code,
   empty branches, leftover experiments or stale names.
6. The change builds what the task actually asked for, not something that only
   looks like it from the outside.

## Reporting

Say what changed, what you ran and what it showed, the net line change, and
anything you could not verify. Report failures and skipped checks plainly; never
claim a check passed that you did not run.

## Commits

- Commit messages are a short imperative sentence in sentence case, no prefix.
- Never add `Co-Authored-By`, "Generated with" or any other AI attribution to
  commits or pull requests.
- Don't commit, push, tag or release unless asked. Releasing is described in
  `CONTRIBUTING.md` and is done by the maintainer.
