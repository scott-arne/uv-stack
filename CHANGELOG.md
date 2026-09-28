# Changelog

Notable changes to uv-stack are documented in this file, beginning with 0.6.0.
Earlier releases predate it.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with
one addition, a `Breaking` section for changes that alter existing behavior, and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.6.0 - 2026-09-28

### Breaking

- `stack show env|profile|bundle NAME` and `stack status NAMES...` now apply
  the same file-stem rule the create and edit commands apply, so a NAME holding
  a path separator, a `.` or `..` segment, `:`, `@`, whitespace, or a leading
  `-` is refused instead of being joined onto the config root and read. A
  hand-made directory carrying such a name is still listed by `stack list env`
  and by a bare `stack status`; naming it on the command line is what stopped
  working, and renaming the directory on disk is what makes it nameable again.
- `stack upgrade --no-upgrade` and `stack upgrade --upgrade-package PKG` now
  preserve the pins the published lock already holds. Both previously compiled
  into an empty file, and `uv pip compile` reads prior pins out of its output
  file, so either flag silently re-resolved every distribution.
- A config file that is a directory, a dangling symlink, or a FIFO is now a
  ConfigError instead of reading as absent. This covers `project-python.txt`,
  `editor.txt`, and an environment's `python.txt`, `micromamba.txt`, and
  `channels.txt`, each of which previously fell back to its default. For all but
  `python.txt` that happened while `stack doctor` reported the root clean; a
  non-regular `python.txt` was reported, but as missing (see Fixed).
- `[tool.uv-stack].python` refuses an empty or space-padded value. An empty
  value previously read as "no preference" and fell through to the machine
  default; a padded one defeated the version test and was taken to name a
  micromamba environment.
- A requirement entry that spans more than one line or ends in a trailing
  backslash is refused. Both shapes were previously written and rendered into
  `requirements.in` verbatim, where a continuation swallows the requirement
  written after it. `stack create env|profile|bundle`, `stack init`, and
  `stack edit` refuse one on the way in; because rendering validates too, every
  command that renders — `upgrade`, `sync`, `status`, `show env`, `refresh`,
  `create project` — now also refuses one already sitting in a `stack.txt` or a
  profile.

### Added

- `stack sync` creates, builds, and recompiles every environment the config
  root declares, without a prompt and creating missing environments rather
  than reporting them as errors, which makes it the first command to run on a
  freshly cloned config root. `stack sync env NAME...` does the same for only
  the named environments. Both accept `--dry-run`, `--stop-on-error`,
  `--strict`, and `--upgrade`.
- `stack sync project TOKENS...` adds tokens to the tracked project in the
  current directory and re-resolves it. The tokens are appended to the stack
  the project already records, never replacing it, so an existing token is not
  lost by forgetting to repeat it. The run goes through the same interrupted-run
  recovery as `stack refresh` and accepts the same `--python`, `--strict`,
  `--no-sync`, and `--dry-run` flags.
- `stack diff SOURCE SOURCE` compares two environments across the four layers
  uv-stack records: the interpreter, the micromamba packages, the effective
  channel order, and the compiled pins. A source is an environment in this
  config root, a copy of another machine's `envs/<name>/` directory, or a bare
  `requirements.lock.txt`, which carries pins only and is reported as such
  rather than as matching. It exits 0 on a difference unless given
  `--exit-code`, and `--json` emits the comparison for scripts.
- Config roots can declare variables in `variables.txt`, supply this machine's
  values in `variables.local.txt` or in an exported environment variable of the
  same name, and reference them as `${NAME}` in profiles, bundles, and
  `stack.txt`. References are expanded when generating `requirements.in` and
  when calling `uv add`, but a tracked project's ledger keeps the unexpanded
  entry so the project travels.
- `stack config portable` writes a managed `.gitignore` block so a config root
  can be committed and cloned. It never invokes git itself: it prints the
  commands for you to run, and `--dry-run` prints the plan without writing.
- `stack create project` and `stack refresh` warn when the interpreter spec they
  record will not resolve on another machine — either an absolute path, or a
  micromamba environment that exists only here.
- `stack doctor` gained diagnostics for sources it cannot read or parse,
  missing editable checkouts, unsafe variable expansion, and a portable
  `.gitignore` block that is absent or stale.
- `stack doctor` warns when a name is published as both a profile and a
  bundle. Resolution is unchanged and still documented -- bare reaches the
  profile, `@<name>` the bundle -- but the only report of the collision used
  to come from the resolver, on the runs that happened to reference the name.
  It is a warning rather than an error, and `--fix` does not touch it: both
  files are valid, and choosing which to withdraw is not doctor's to make.

### Changed

- `stack show env` prints a note and exits 0 when `requirements.in` cannot be
  rendered, matching what `stack status` already did. It previously printed the
  whole description and then exited 1, while `--json` exited 0 on the same
  root.
- The upgrade and sync batch summary reports three outcomes rather than
  two: succeeded, failed, and skipped. A `--dry-run` batch prints the summary
  too, where it previously printed none and still exited 1 when an environment
  failed.
- The package declares POSIX support in its classifiers. `fcntl`, `pty`,
  `termios`, `O_NOFOLLOW`, and `O_NONBLOCK` are all absent on Windows, where
  `stack config portable` can create an absent ignore file but cannot refresh
  an existing one.

### Fixed

- `stack upgrade` publishes `requirements.lock.txt` with the mode every other
  generated file gets rather than `mkstemp`'s 0600, which on a config root
  shared with a second account left the lock the one generated file they could
  not read.
- `stack upgrade NAME NAME` upgrades that environment once instead of twice.
- A filesystem error in one environment no longer aborts the whole batch; it is
  reported in the summary like any other failure.
- An environment that `--stop-on-error` abandoned after an earlier failure is
  reported as skipped rather than as a success.
- `stack doctor` no longer answers for a path it could not read. An unsearchable
  config root previously turned `profiles/`, `bundles/`, and `envs/` into
  "missing" errors offering a `mkdir` that would fail for the same reason the
  stat did, and a top-level directory it could not enumerate produced no finding
  at all.
- `stack doctor` no longer calls a `python.txt` that is present but not a
  regular file missing, a report whose offered repair could not run because the
  entry was already there.
- `stack doctor` reports an environment whose `stack.txt` is not a regular file.
  Such an environment is dropped from every listing in the program, and doctor
  skipped it on the same probe and printed "No problems detected."
- `stack doctor` reports an env-like directory under `envs/` that has no
  `stack.txt` at all -- one holding a `requirements.in` or an `environment.yml`,
  so a sync ran there. It is dropped from every listing for the same reason,
  which left a bare `stack sync` passing over it without a word -- or, with no
  other environment declared, answering that there were none -- while its own
  compiled lock sat beside it. A directory under `envs/` with no
  such marker is still not reported: it is not a broken environment, it is not
  an environment.
- An unreadable `bundles/` no longer fails a command that resolved a bare
  profile token. The shadow warning -- the one saying a name matches both a
  profile and a bundle -- probed `bundles/<name>.yaml` to decide whether to
  print, and that probe raises rather than answering no when the directory
  cannot be searched, so `stack status`, `stack show env`, and `stack upgrade`
  exited 1 over a directory none of them needed to read. The warning is now
  skipped when the probe cannot answer; the resolution it decorated is
  unchanged, and `stack doctor` still reports the directory.
- Every YAML loader failure on a profile or bundle becomes a ConfigError. Only
  `YAMLError` was converted before, so a document whose constructor raises
  something else -- a date of `2020-99-99`, or `!!bool "nope"` -- came out of
  `stack doctor` as a traceback.
- `stack create project` resuming an interrupted run keeps editable, VCS, and
  path entries in `tracking.applied` instead of deleting them, which silently
  handed ownership of those dependencies back to the user.
- An environment marker no longer hides a requirement's name from ownership
  tracking. A `/` or `\` anywhere in the entry, including inside the marker,
  made it read as a path and so as user-owned.
- Runs that rewrite a tracked project's `[tool.uv-stack]` table -- `stack
  refresh`, `stack sync project`, and `stack create project` -- take a
  per-project lock, so a second run in the same project waits up to five
  seconds and is then refused. Unserialized, the last run to write the table
  replaced it with the view it had read at the start, so two `stack sync
  project` runs could each drop the other's token. `--dry-run` does not wait.
  Where the config root cannot lock, `stack doctor` says so and these runs stay
  unserialized.

### Security

- `stack upgrade` validates every NAME before it starts the batch, so a name
  that would escape the config root is refused rather than used to read another
  directory's sources and overwrite its generated files.
- The guarded read behind `stack config portable` — a command new in this
  release, so no published version was exposed — refuses an existing
  `.gitignore` that is a symlink or that carries a second hard link, and
  refuses outright on a platform without `O_NOFOLLOW` and `O_NONBLOCK` rather
  than degrading to an unguarded open. A refresh preserves everything outside
  the managed block byte for byte, so each route would otherwise copy a planted
  file's contents into the `.gitignore` you go on to commit.
