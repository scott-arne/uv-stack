# uv-stack

A command-line interface and library that formalizes a `uv` + `micromamba`
Python environment workflow built around profiles, bundles, stacks, and named
environments.

```bash
pip install uv-stack
```
Usage:

```text
$ stack --help

 Usage: stack [OPTIONS] COMMAND [ARGS]...                                       
                                                                                
 stack — formalized uv + micromamba environment management.                     
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --version        Show the version and exit.                                  │
│ --root     TEXT  Config root. Defaults to $UV_ENV_ROOT or                    │
│                  ~/.config/python-envs.                                      │
│ --help           Show this message and exit.                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Environments ───────────────────────────────────────────────────────────────╮
│ upgrade          Render, compile, install, and check one or more existing    │
│                  environments.                                               │
│ create           Create a new environment or project.                        │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Inspection ─────────────────────────────────────────────────────────────────╮
│ list             List resources of KIND (env, profile, or bundle).           │
│ show             Show details of KIND NAME. NAME defaults to ``main`` for    │
│                  ``env``.                                                    │
│ resolve          Resolve TOKENS and print the result.                        │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Maintenance ────────────────────────────────────────────────────────────────╮
│ doctor           Detect layout/config problems and print suggested fixes.    │
│ config           Initialize the config tree.                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```
