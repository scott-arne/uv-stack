"""Resolve stack tokens into an ordered, de-duplicated :class:`ResolvedStack`.

The grammar mirrors the original zsh tooling:

``@x`` / ``bundle:x``
    Resolve bundle ``x`` (recursively resolving each of its lines).
``profile:x``
    Add profile ``x`` (must exist).
``package:x`` / ``pkg:x``
    Add ``x`` as a literal inline requirement.
unqualified ``x``
    Profile if ``profiles/x.in`` exists, else bundle if ``bundles/x.bundle``
    exists, else a literal inline requirement.
``-e <path>`` / archive paths / anything else
    Literal inline requirement.

Profiles are recorded by name (rendered later as ``-r profiles/<name>.in``).
First occurrence wins for ordering; bundles already on the resolution path are
skipped, so mutually-referential bundles cannot recurse forever.
"""

from __future__ import annotations

from collections.abc import Iterable

from uv_stack.config import ConfigRoot
from uv_stack.errors import ResolutionError
from uv_stack.models import ResolvedStack


class Resolver:
    """Turns stack tokens into a :class:`ResolvedStack`.

    :param config: The configuration root used to look up profiles and bundles.
    """

    def __init__(self, config: ConfigRoot) -> None:
        self._config = config

    def resolve(self, tokens: Iterable[str]) -> ResolvedStack:
        """Resolve ``tokens`` into profiles and inline requirements.

        :param tokens: Stack tokens (from a ``stack.txt``, a bundle, or the CLI).
        :returns: The resolved, de-duplicated stack.
        :raises ResolutionError: For an explicitly-qualified profile or bundle
            that does not exist.
        """
        self._profiles: list[str] = []
        self._inline: list[str] = []
        self._seen_profiles: set[str] = set()
        self._seen_inline: set[str] = set()
        self._seen_bundles: set[str] = set()
        for token in tokens:
            self._resolve_token(token)
        return ResolvedStack(profiles=self._profiles, inline=self._inline)

    def classify(self, tokens: Iterable[str]) -> list[str]:
        """Classify each token without expanding bundles or profiles.

        Each token is labeled by what it *is* — ``bundle:<name>``,
        ``profile:<name>``, or ``package:<spec>`` — using the same precedence as
        :meth:`resolve` but with no recursion. The result is de-duplicated with
        first occurrence winning.

        :param tokens: Stack tokens to classify.
        :returns: Full specifiers (e.g. ``["bundle:standard", "package:numpy"]``).
        """
        classified: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            spec = self._classify_token(token)
            if spec is not None and spec not in seen:
                seen.add(spec)
                classified.append(spec)
        return classified

    def resolve_packages(self, tokens: Iterable[str]) -> list[str]:
        """Fully resolve ``tokens`` to a flat, de-duplicated package list.

        Bundles and profiles are expanded to the packages they contain, so the
        result is a raw requirements list with no ``-r`` references or kind
        prefixes (compatible with a ``requirements.txt``). Profile packages come
        first in resolution order, followed by inline requirements.

        :param tokens: Stack tokens to resolve.
        :returns: Package specifiers in order, de-duplicated.
        :raises ResolutionError: For an explicitly-qualified profile or bundle
            that does not exist.
        """
        stack = self.resolve(tokens)
        packages: list[str] = []
        seen: set[str] = set()
        for name in stack.profiles:
            for req in self._config.load_profile(name).requirements:
                if req not in seen:
                    seen.add(req)
                    packages.append(req)
        for req in stack.inline:
            if req not in seen:
                seen.add(req)
                packages.append(req)
        return packages

    def _classify_token(self, token: str) -> str | None:
        token = token.strip()
        if not token:
            return None
        if token.startswith("@"):
            return f"bundle:{token[1:]}"
        if token.startswith("bundle:"):
            return token
        if token.startswith("profile:"):
            return token
        if token.startswith("package:"):
            return token
        if token.startswith("pkg:"):
            return f"package:{token[len('pkg:') :]}"
        if self._config.profile_exists(token):
            return f"profile:{token}"
        if self._config.bundle_exists(token):
            return f"bundle:{token}"
        return f"package:{token}"

    def _resolve_token(self, token: str) -> None:
        token = token.strip()
        if not token:
            return

        if token.startswith("@"):
            self._resolve_bundle(token[1:], explicit=True)
        elif token.startswith("bundle:"):
            self._resolve_bundle(token[len("bundle:") :], explicit=True)
        elif token.startswith("profile:"):
            self._add_profile(token[len("profile:") :], explicit=True)
        elif token.startswith("package:"):
            self._add_inline(token[len("package:") :])
        elif token.startswith("pkg:"):
            self._add_inline(token[len("pkg:") :])
        elif self._config.profile_exists(token):
            self._add_profile(token, explicit=False)
        elif self._config.bundle_exists(token):
            self._resolve_bundle(token, explicit=False)
        else:
            self._add_inline(token)

    def _add_profile(self, name: str, *, explicit: bool) -> None:
        if explicit and not self._config.profile_exists(name):
            raise ResolutionError(
                f"Missing profile: {self._config.profile_path(name)}",
                hint="Check the profile name or create the .in file.",
            )
        if name not in self._seen_profiles:
            self._seen_profiles.add(name)
            self._profiles.append(name)

    def _add_inline(self, requirement: str) -> None:
        requirement = requirement.strip()
        if requirement and requirement not in self._seen_inline:
            self._seen_inline.add(requirement)
            self._inline.append(requirement)

    def _resolve_bundle(self, name: str, *, explicit: bool) -> None:
        if explicit and not self._config.bundle_exists(name):
            raise ResolutionError(
                f"Missing bundle: {self._config.bundle_path(name)}",
                hint="Check the bundle name or create the .bundle file.",
            )
        if name in self._seen_bundles:
            return
        self._seen_bundles.add(name)
        bundle = self._config.load_bundle(name)
        for token in bundle.tokens:
            self._resolve_token(token)
