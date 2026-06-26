# Contributing to FileManager (Triage Explorer)

Thanks for your interest in contributing. This document describes how to set up
the project, the git workflow we follow, and the conventions your changes should
respect.

By contributing you agree that your work is licensed under the project's
[GPL-3.0 license](LICENSE).

## Project layout

| Path        | What it is                                                        |
| ----------- | ----------------------------------------------------------------- |
| `research/` | Python sandbox: embedding engines, evaluation tasks, benchmarks   |
| `frontend/` | Desktop UI (Avalonia 12 on .NET 10, C#)                           |
| `docs/`     | Design specs and documentation                                    |

Most changes touch a single component. Keep a pull request scoped to one of them
unless a change genuinely spans both.

## Development setup

### Research (Python)

```bash
cd research
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
pytest                       # run the test suite
```

### Frontend (.NET)

Requires the .NET 10 SDK.

```bash
cd frontend
dotnet restore
dotnet build
dotnet run
```

## Git workflow

We keep history linear and readable. Please follow these practices:

1. **Never commit directly to `main`.** Branch off the latest `main`:

   ```bash
   git switch main
   git pull --rebase origin main
   git switch -c feat/short-description
   ```

2. **Name branches by type and scope**, for example:
   `feat/clap-audio-engine`, `fix/loader-empty-doc`, `docs/contributing`.

3. **Keep commits atomic.** Each commit should be one self-contained, working
   change. Don't mix unrelated changes. Don't commit half-finished work.

4. **Rebase, don't merge.** Keep your branch up to date with
   `git pull --rebase origin main`. This avoids noisy merge commits.

5. **Force-push only your own feature branch** (`git push --force-with-lease`),
   never a shared branch.

6. Run the relevant test suite and make sure it passes before opening a PR.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/). Format:

```
<type>: <imperative summary, ~50 chars>

<optional body explaining what and why, wrapped at ~72 chars>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`.

Rules:

- Use the imperative mood: "add CLAP engine", not "added" or "adds".
- Keep the summary under ~50 characters and don't end it with a period.
- Explain the *why* in the body when it isn't obvious from the diff.
- Reference issues in the body, for example `Closes #12`.

Example:

```
feat: add CLAP audio embedding engine

CLAP gives a shared text-audio space, so a text query can retrieve
audio clips directly. Benchmarked against the MFCC baseline on ESC-50.
```

## Code style

### Python (`research/`)

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Docstrings use compact numpydoc: a section header underlined with `---`
  (three dashes), and one-line `name : description` entries with single spaces,
  no colon-alignment padding:

  ```python
  """Engine summary.

  Parameters
  ---
  model_name : HuggingFace model id (default ...)
  batch_size : number of items per forward pass (default 32)
  """
  ```

- Don't add decorative comment dividers (no `# ---- Section ----`). A short
  `# Section` comment is enough.
- New engines, tasks, and loaders should have a focused test under
  `research/tests/`.

### C# (`frontend/`)

- Follow the standard .NET / C# conventions (PascalCase for types and members,
  `_camelCase` for private fields).
- Keep `dotnet build` warning-free.

## Pull requests

1. Rebase onto the latest `main` and ensure history is clean (squash fixup
   commits).
2. Make sure tests pass and the build succeeds.
3. Open the PR against `main` with a clear description of what changed and why.
   Link any related issues.
4. Be ready to address review feedback by amending or adding commits and
   force-pushing your branch.

## Reporting issues

Open an issue describing the problem, the steps to reproduce it, and what you
expected. Include your OS and the component (`research` or `frontend`).
