"""Resolve stack tokens into an ordered, de-duplicated :class:`ResolvedStack`.

The grammar mirrors the original zsh tooling:

``@x`` / ``bundle:x``
    Resolve bundle ``x`` (recursively resolving each of its lines).
``profile:x``
    Add profile ``x`` (must exist).
``package:x`` / ``pkg:x``
    Add ``x`` as a literal inline requirement.
unqualified ``x``
    Profile if ``profiles/x.yaml`` exists, else bundle if ``bundles/x.yaml``
    exists, else a literal inline requirement.
``-e <path>`` / archive paths / anything else
    Literal inline requirement.

Profiles are recorded by name (expanded inline at render time). First
occurrence wins for ordering; a bundle already on the active resolution path
is skipped with a warning, so mutually-referential bundles cannot recurse
forever and cannot fail silently either.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable

from uv_stack.config import ConfigRoot
from uv_stack.errors import ResolutionError
from uv_stack.models import ClassifiedTokens, ResolvedStack

#: Tokens that look like a plain profile/bundle/package name. Anything with a
#: version specifier, path separator, extras bracket, or flag is clearly a
#: requirement, so strict mode and near-miss checks leave it alone. Leading '-'
#: (flags) and leading '.' (dot paths) are excluded.
_PLAIN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def bundle_self_references(name: str, tokens: Iterable[str]) -> list[str]:
    """Tokens that reference bundle ``name`` itself.

    A bundle whose includes name the bundle contributes nothing: the recursion
    guard skips the repeat visit, so the token silently expands to an empty
    list. A bare token is the dangerous form — at create time the bundle does
    not exist yet, so it classifies as a literal package and passes every
    existence check; the moment the file lands it becomes a self-reference.

    :param name: The bundle being created.
    :param tokens: The tokens destined for its ``includes``.
    :returns: The offending tokens, in input order.
    """
    hits: list[str] = []
    for raw in tokens:
        token = raw.strip()
        if token.startswith("@"):
            referenced = token[1:]
        elif token.startswith("bundle:"):
            referenced = token[len("bundle:") :]
        elif _PLAIN_NAME_RE.match(token):
            referenced = token
        else:
            continue
        if referenced == name:
            hits.append(raw)
    return hits


class Resolver:
    """Turns stack tokens into a :class:`ResolvedStack`.

    :param config: The configuration root used to look up profiles and bundles.
    :param strict: When true, an unqualified plain-name token that falls
        through to a literal package raises :class:`ResolutionError` instead of
        resolving silently.
    """

    def __init__(self, config: ConfigRoot, *, strict: bool = False) -> None:
        self._config = config
        self._strict = strict

    def resolve(self, tokens: Iterable[str]) -> ResolvedStack:
        """Resolve ``tokens`` into profiles and inline requirements.

        :param tokens: Stack tokens (from a ``stack.txt``, a bundle, or the CLI).
        :returns: The resolved, de-duplicated stack (with any warnings).
        :raises ResolutionError: For an explicitly-qualified profile or bundle
            that does not exist, or a bare literal fallthrough in strict mode.
        """
        self._profiles: list[str] = []
        self._inline: list[str] = []
        self._warnings: list[str] = []
        self._seen_profiles: set[str] = set()
        self._seen_inline: set[str] = set()
        self._seen_bundles: set[str] = set()
        self._bundle_stack: list[str] = []
        for token in tokens:
            self._resolve_token(token)
        return ResolvedStack(
            profiles=self._profiles,
            inline=self._inline,
            warnings=list(dict.fromkeys(self._warnings)),
        )

    def classify(self, tokens: Iterable[str]) -> ClassifiedTokens:
        """Classify each token without expanding bundles or profiles.

        Each token is labeled by what it *is* — ``bundle:<name>``,
        ``profile:<name>``, or ``package:<spec>`` — using the same precedence as
        :meth:`resolve` but with no recursion. Entries are de-duplicated with
        first occurrence winning; the same strict/warning rules as
        :meth:`resolve` apply to bare tokens.

        :param tokens: Stack tokens to classify.
        :returns: The classified entries plus any warnings.
        :raises ResolutionError: For a bare literal fallthrough in strict mode.
        """
        self._warnings = []
        classified: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            spec = self._classify_token(token)
            if spec is not None and spec not in seen:
                seen.add(spec)
                classified.append(spec)
        return ClassifiedTokens(
            entries=classified, warnings=list(dict.fromkeys(self._warnings))
        )

    def flatten(self, stack: ResolvedStack) -> list[str]:
        """Flatten an already-resolved stack to a de-duplicated package list.

        Profile packages come first in resolution order, then inline
        requirements — the same shape :meth:`resolve_packages` returns.

        :param stack: A stack previously returned by :meth:`resolve`.
        :returns: Package specifiers in order, de-duplicated.
        """
        packages: list[str] = []
        seen: set[str] = set()
        for name in stack.profiles:
            for req in self._config.load_profile(name).includes:
                if req not in seen:
                    seen.add(req)
                    packages.append(req)
        for req in stack.inline:
            if req not in seen:
                seen.add(req)
                packages.append(req)
        return packages

    def resolve_packages(self, tokens: Iterable[str]) -> list[str]:
        """Fully resolve ``tokens`` to a flat, de-duplicated package list.

        :param tokens: Stack tokens to resolve.
        :returns: Package specifiers in order, de-duplicated.
        :raises ResolutionError: For an explicitly-qualified profile or bundle
            that does not exist, or a bare literal fallthrough in strict mode.
        """
        return self.flatten(self.resolve(tokens))

    # -- shared checks -----------------------------------------------------

    def _check_bare_literal(self, token: str) -> None:
        """Apply strict/near-miss rules to a bare token that became a literal."""
        if not _PLAIN_NAME_RE.match(token):
            return
        if self._strict:
            raise ResolutionError(
                f"Unqualified token '{token}' resolved to a literal package.",
                hint=(
                    f"Use pkg:{token} for a literal package, or fix the "
                    "profile/bundle name."
                ),
            )
        known = sorted(
            set(self._config.list_profiles()) | set(self._config.list_bundles())
        )
        matches = difflib.get_close_matches(token, known, n=1, cutoff=0.8)
        if matches:
            self._warnings.append(
                f"'{token}' resolved to a literal package; did you mean "
                f"'{matches[0]}'? (use pkg:{token} to silence)"
            )

    def _warn_shadow(self, token: str) -> None:
        if self._config.bundle_exists(token):
            self._warnings.append(
                f"'{token}' matches both a profile and a bundle; using the "
                f"profile (use @{token} for the bundle)"
            )

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
            self._warn_shadow(token)
            return f"profile:{token}"
        if self._config.bundle_exists(token):
            return f"bundle:{token}"
        self._check_bare_literal(token)
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
            self._warn_shadow(token)
            self._add_profile(token, explicit=False)
        elif self._config.bundle_exists(token):
            self._resolve_bundle(token, explicit=False)
        else:
            self._check_bare_literal(token)
            self._add_inline(token)

    def _add_profile(self, name: str, *, explicit: bool) -> None:
        if explicit and not self._config.profile_exists(name):
            raise ResolutionError(
                f"Missing profile: {self._config.profile_path(name)}",
                hint="Check the profile name or create the .yaml file.",
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
                hint="Check the bundle name or create the .yaml file.",
            )
        # A bundle already on the ACTIVE path is a cycle: the reference cannot
        # contribute anything, and staying silent is how a self-referencing
        # bundle came to look like it worked. A bundle merely already SEEN is a
        # diamond — legitimate, common, and deliberately silent.
        if name in self._bundle_stack:
            cycle = " -> ".join([*self._bundle_stack, name])
            self._warnings.append(
                f"Bundle cycle skipped: {cycle}. The repeated reference "
                f"contributes nothing; use pkg:{name} if you meant the "
                "literal package."
            )
            return
        if name in self._seen_bundles:
            return
        self._seen_bundles.add(name)
        self._bundle_stack.append(name)
        try:
            bundle = self._config.load_bundle(name)
            for token in bundle.includes:
                self._resolve_token(token)
        finally:
            self._bundle_stack.pop()
