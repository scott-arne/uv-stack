# import json
# import re
# import shutil
import sys
from pathlib import Path

# noinspection PyPackageRequirements
from invoke.tasks import task

ROOT = Path(__file__).parent.absolute()

@task
def test(c):
    """Run the test suite with pytest"""
    c.run(f"{sys.executable} -m pytest tests/")


@task
def build(c):
    """Build distribution packages"""
    c.run("rm -rf dist")
    c.run(f"{sys.executable} -m build")


@task
def upload(c):
    """Upload package to PyPI (requires PyPI credentials configured)"""
    c.run("rm -rf dist")
    c.run(f"{sys.executable} -m build")
    c.run(f"{sys.executable} -m twine upload dist/*")


@task
def publish(c):
    c.run(f"cd {ROOT} && rm -rf dist/ && {sys.executable} -m build --wheel && {sys.executable} -m twine upload dist/*")


# @task
# def docs(c):
#     """Build Sphinx documentation"""
#     docs_dir = ROOT / "docs"
#     build_dir = docs_dir / "_build"
#     c.run(f"cd {docs_dir} && {sys.executable} -m sphinx -b html . {build_dir}/html")


# @task
# def serve_docs(c, port=8000, watch=False):
#     """Serve Sphinx documentation locally.

#     :param port: Port to serve on (default: 8000).
#     :param watch: Watch for changes and auto-rebuild (requires sphinx-autobuild).
#     """
#     docs_dir = ROOT / "docs"
#     html_dir = docs_dir / "_build" / "html"

#     if watch:
#         # sphinx-autobuild performs its own initial build, so skip the pre-build.
#         print(f"Watching for changes and serving docs at http://localhost:{port}")
#         c.run(
#             f"{sys.executable} -u -m sphinx_autobuild {docs_dir} {html_dir} --port {port}",
#             pty=True,
#         )
#     else:
#         if not html_dir.exists():
#             print("Documentation not built. Building first...")
#             docs(c)
#         print(f"Serving docs at http://localhost:{port}")
#         c.run(f"cd {html_dir} && {sys.executable} -u -m http.server {port}", pty=True)
