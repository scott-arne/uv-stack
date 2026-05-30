from __future__ import annotations

from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot


@pytest.fixture
def config_tree(tmp_path: Path) -> ConfigRoot:
    """A minimal but realistic config tree under tmp_path.

    profiles: ds, chem, utils
    bundles:  standard (ds chem utils), qsar (standard + umap-learn)
    envs:     main (stack: @standard, python 3.12, micromamba: graphviz)
    """
    root = tmp_path / "python-envs"
    (root / "profiles").mkdir(parents=True)
    (root / "bundles").mkdir(parents=True)
    (root / "envs" / "main").mkdir(parents=True)

    (root / "profiles" / "ds.in").write_text("numpy\npandas\n")
    (root / "profiles" / "chem.in").write_text("rdkit\n")
    (root / "profiles" / "utils.in").write_text("rich\n")

    (root / "bundles" / "standard.bundle").write_text("ds\nchem\nutils\n")
    (root / "bundles" / "qsar.bundle").write_text("standard\numap-learn\n")

    env = root / "envs" / "main"
    env.joinpath("python.txt").write_text("3.12\n")
    env.joinpath("stack.txt").write_text("@standard\n")
    env.joinpath("micromamba.txt").write_text("graphviz\n")
    env.joinpath("requirements.local.in").write_text("some-local-pkg\n")

    return ConfigRoot(root)
