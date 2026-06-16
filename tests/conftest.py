from __future__ import annotations

from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot


@pytest.fixture
def config_tree(tmp_path: Path) -> ConfigRoot:
    """A minimal but realistic config tree under tmp_path.

    profiles: ds, chem, utils
    bundles:  standard (ds chem utils), qsar (standard + umap-learn)
    envs:     main (stack: @standard, python 3.12, micromamba: graphviz,
              channels: bioconda)
    """
    root = tmp_path / "python-envs"
    (root / "profiles").mkdir(parents=True)
    (root / "bundles").mkdir(parents=True)
    (root / "envs" / "main").mkdir(parents=True)

    (root / "profiles" / "ds.yaml").write_text(
        "description: Core data-science stack\n"
        "tags: [data, core]\n"
        "includes:\n  - numpy\n  - pandas\n"
    )
    (root / "profiles" / "chem.yaml").write_text(
        "description: Cheminformatics\nincludes:\n  - rdkit\n"
    )
    (root / "profiles" / "utils.yaml").write_text("includes:\n  - rich\n")

    (root / "bundles" / "standard.yaml").write_text(
        "description: Everything for daily work\n"
        "includes:\n  - ds\n  - chem\n  - utils\n"
    )
    (root / "bundles" / "qsar.yaml").write_text(
        "includes:\n  - standard\n  - umap-learn\n"
    )

    env = root / "envs" / "main"
    env.joinpath("python.txt").write_text("3.12\n")
    env.joinpath("stack.txt").write_text("@standard\n")
    env.joinpath("micromamba.txt").write_text("graphviz\n")
    env.joinpath("channels.txt").write_text("bioconda\n")
    env.joinpath("requirements.local.in").write_text("some-local-pkg\n")

    return ConfigRoot(root)
