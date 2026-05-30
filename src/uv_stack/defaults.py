"""Default profile and bundle contents seeded by ``stack config init``.

Each value is the full file text. Contents mirror the original workflow
specification. ``config init`` writes only files that do not already exist.
"""

from __future__ import annotations

DEFAULT_PROFILES: dict[str, str] = {
    "ds": (
        "numpy\nscipy\nscipy-stubs\npandas\npandas-stubs\npolars\npyarrow\n"
        "pyarrow-stubs\nscikit-learn\nnetworkx\nduckdb\nnumba\n"
    ),
    "chem": "rdkit\nbiopython\ngemmi\nmmpdb\n",
    "openeye": "openeye-toolkits\nopeneye-orionplatform\n",
    "marimo": "marimo\n",
    "jupyter": "ipykernel\njupyterlab\nipywidgets\nnotebook\n",
    "viz": "matplotlib\nrustworkx\nseaborn\nplotly\nbokeh\naltair\n",
    "utils": (
        "attrs\ntqdm\nrich\nrich-click\nclick\nclick-option-group\ntyper\n"
        "pydantic\nrequests\nhttpx\npython-dotenv\nast_serialize\nasttokens\n"
        "beautifulsoup4\ndill\ninvoke\nmore-itertools\nmultimethod\nmultiprocess\n"
        "multiprocess-stubs\npsutil\nPyJWT\nrequests-toolbelt\nruamel.yaml\n"
        "setuptools\nsortedcontainers\nsortedcontainers-stubs\ntabulate\nxlrd\n"
        "xlsxwriter\n"
    ),
    "db-cloud": "boto3\nboto3-stubs\npsycopg\nSQLAlchemy\n",
    "web": "fastapi\nuvicorn\n",
    "build": "build\ncookiecutter\ndelocate\nscikit_build_core\ntwine\n",
    "docs": "Sphinx\nsphinx-autobuild\n",
    "dev": "ruff\npytest\nmypy\n",
    "livedesign": (
        "/Users/johnss51/Support/schrodinger/ldclient/packages/ldclient-2024.1.4.tar.gz\n"
    ),
    "local-editable": (
        "-e /Users/johnss51/Development/python/vrzn\n"
        "-e /Users/johnss51/Development/python/simple-run-model\n"
        "-e /Users/johnss51/Development/cpp/oeselect/python\n"
        "\n"
        "-e /Users/johnss51/Development/cpp/oemaestro/python\n"
        "-e /Users/johnss51/Development/python/oepandas\n"
        "-e /Users/johnss51/Development/python/oepolars\n"
        "-e /Users/johnss51/Development/python/oepandas-mae\n"
        "-e /Users/johnss51/Development/python/oepolars-mae\n"
        "-e /Users/johnss51/Development/cpp/oeio/python\n"
        "\n"
        "-e /Users/johnss51/Development/python/cnotebook\n"
        "\n"
        "-e /Users/johnss51/Development/python/somf\n"
        "-e /Users/johnss51/Development/python/oeconformity\n"
        "-e /Users/johnss51/Development/cpp/maptitude/python\n"
        "-e /Users/johnss51/Development/cpp/oefp/python\n"
        "-e /Users/johnss51/Development/cpp/oemmpa/python\n"
        "-e /Users/johnss51/Development/python/pydinger\n"
        "-e /Users/johnss51/Development/python/bms-chem\n"
        "-e /Users/johnss51/Development/python/bms-bio\n"
        "-e /Users/johnss51/Development/python/posit-mmgbsa\n"
        "\n"
        "-e /Users/johnss51/Development/python/sqlalchemy-bms\n"
        "-e /Users/johnss51/Development/python/cadd-scripts\n"
    ),
}

DEFAULT_BUNDLES: dict[str, str] = {
    "minimal": "ds\nutils\n",
    "standard": "ds\nchem\nviz\nutils\n",
    "notebook": "ds\nchem\nviz\nutils\nmarimo\njupyter\n",
    "openeye": "ds\nchem\nopeneye\nviz\nutils\nmarimo\njupyter\n\ndatamol\nmols2grid\n",
    "full": (
        "ds\nchem\nopeneye\nviz\nmarimo\njupyter\nutils\ndb-cloud\nweb\nbuild\n"
        "docs\ndev\nlivedesign\nlocal-editable\n"
    ),
    "chemprop": "ds\nviz\nmarimo\njupyter\nutils\ndev\n",
    "qsar": "ds\nchem\nviz\nmarimo\nutils\n\numap-learn\nhdbscan\ndatamol\nmols2grid\n",
}
