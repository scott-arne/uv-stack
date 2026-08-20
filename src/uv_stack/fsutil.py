"""Filesystem helpers shared by operations."""

from __future__ import annotations

import errno
import io
import os
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Literal, NamedTuple

from uv_stack.errors import ConfigError

#: os.link failures that mean "this filesystem cannot hard-link" — fall back
#: to exclusive create. EEXIST is deliberately absent: that is the no-clobber
#: contract, never a fallback trigger.
_LINK_FALLBACK_ERRNOS = frozenset(
    {errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK}
    | ({getattr(errno, "ENOTSUP")} if hasattr(errno, "ENOTSUP") else set())  # noqa: B009
)

#: True only where BOTH guard flags exist. Opening a target whose type is not
#: known in advance needs both: O_NOFOLLOW so a symlinked target is never
#: opened, O_NONBLOCK so a FIFO target never blocks the open. Where either is
#: absent getattr yields 0, which does not weaken the guard — it removes it,
#: and an identity recheck afterwards comes too late to make up for it: it can
#: reject what the open returned, not stop the open from following a symlink
#: or hanging on a FIFO. Every caller that can decline the open therefore
#: skips it entirely rather than take it unguarded; what each gives up by
#: skipping differs, and is stated at the call site. name_lock's create is the
#: one open that cannot be declined — there is no lock without it — so it
#: passes both constants for whatever they are worth on the platform and
#: rejects a non-regular target after the fact.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_FASTPATH_AVAILABLE = bool(_O_NOFOLLOW) and bool(_O_NONBLOCK)


def _lock_supported() -> bool:
    """Whether this platform has the POSIX advisory locking ``name_lock`` needs."""
    try:
        import fcntl  # noqa: F401
    except ImportError:  # pragma: no cover - non-Unix platforms
        return False
    return True


#: True only where fcntl.flock exists. Where it does not, name_lock is a no-op
#: and the writers fall back to their post-publish collision checks alone: the
#: both-processes-survive race stays closed, while the killed-mid-window case
#: and the adopter residual do not. Same degrade-silently posture as
#: _FASTPATH_AVAILABLE above and _PTY_AVAILABLE in runner.py.
_LOCK_AVAILABLE = _lock_supported()

#: Read at call time, not bound as a default argument, so a test can shorten it.
_LOCK_TIMEOUT = 5.0

#: How long to wait between flock attempts. Small enough that an uncontended
#: handoff is imperceptible, large enough not to spin.
_LOCK_POLL = 0.01

#: Errnos treated as contention — another process holds the lock. EACCES is
#: here because CPython's fcntl.flock falls back to fcntl(F_SETLK) where the
#: build lacks HAVE_FLOCK, and POSIX permits F_SETLK to report a conflicting
#: lock as either EACCES or EAGAIN. Any other errno is treated as "this
#: filesystem cannot lock" and degrades to the no-fcntl no-op: some NFS and
#: FUSE mounts return ENOLCK or EOPNOTSUPP, and the degrade is chosen so an
#: unlockable filesystem keeps working rather than failing every create. The
#: trade-off is that a transient ENOLCK (e.g., kernel out of memory for lock
#: records on Linux) also degrades, where a retry loop might succeed.
_LOCK_CONTENDED_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES})

#: Errnos from the read-only retry that mean "no descriptor because of what
#: this user may do here", as opposed to "no descriptor because of what is
#: sitting here". A .locks directory this user may not write gives ENOENT —
#: there is nothing to open, and nothing may be created — and a path this user
#: is barred from, by permission bits or by a MAC layer, gives EACCES or, on
#: some systems, EPERM. Membership does not decide the outcome on its own:
#: these errnos say nothing about the target's type, so they fall through to
#: the lstat below, which refuses a plant and degrades only where it finds no
#: plant to refuse. An errno outside the set is re-raised, because degrading on
#: it would silently cost exclusion. Little can reach here and not be a
#: permission problem, because the retry runs only after the create raised
#: PermissionError: either something was swapped in between the two opens — a
#: symlink, ELOOP; a socket whose type the kernel will not open, EOPNOTSUPP
#: here and ENXIO elsewhere; a device node with no driver behind it, ENXIO — or
#: the open failed for a reason with nothing to do with the path at all. Those
#: same shapes standing there all along never reach this set: the create
#: refuses them itself, and what it raises is not a PermissionError, so it is
#: never caught.
#:
#: Which side a shape falls on is the platform's ordering rather than a
#: property of the shape: this kernel refuses a socket by type at every mode,
#: while one that checks permission first reports EACCES for the same socket at
#: 0200 and so puts it inside the set. Neither order degrades, which is the
#: point, but they refuse by different routes: a kernel that stops the open by
#: type re-raises what the create returned — above, without this set ever being
#: consulted — while one that stops it on permissions falls through to the
#: lstat below. A directory that was there all along reaches neither route: the
#: create fails EISDIR, which is not a permission problem and not this set.
_LOCK_UNOBTAINABLE_ERRNOS = frozenset({errno.ENOENT, errno.EACCES, errno.EPERM})


def nofollow_read_flags() -> int | None:
    """Open flags for a read that must not follow a symlink or block.

    ``O_NOFOLLOW`` refuses a symlink at the path and ``O_NONBLOCK`` keeps the
    open from blocking on a FIFO. Both are required for the read to be safe,
    so a platform missing either gets ``None`` rather than a weaker flag set —
    the caller declines the read instead of making an unsafe one.

    :returns: Flags for :func:`os.open`, or ``None`` where this platform lacks
        the guards the read needs.
    """
    if not _FASTPATH_AVAILABLE:
        return None
    return os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK


class Published(NamedTuple):
    """How :func:`link_or_copy_no_replace` published a file, and what it saw.

    :ivar kind: ``"link"`` when ``dst`` is a second name for ``src``'s own
        inode, ``"copy"`` when it is a fresh inode holding a copy of the bytes.
        This field, and only this field, tells a caller which mechanism ran.
    :ivar stat: For ``"copy"``, the stat of the inode this call created — an
        exclusive create, so it is proven ours, and it is the identity ``dst``
        was given. For ``"link"``, merely an observation of what ``src`` named
        at the moment this call read it. It is **not** proof of what
        :func:`os.link` published: if ``src`` is replaced between that read and
        the link, the replacement is what appears at ``dst`` and this stat
        describes the inode that no longer holds the name. A caller that must
        reason about identity across a concurrent replacement of ``src`` gets
        no stronger guarantee here than :func:`os.link` itself offers, and must
        keep the same defensive rules it would use around a bare
        :func:`os.link`.
    """

    kind: Literal["link", "copy"]
    stat: os.stat_result


def link_or_copy_no_replace(
    src: Path | str, dst: Path, *, fallback_errnos: frozenset[int] = _LINK_FALLBACK_ERRNOS
) -> Published:
    """Publish the regular file at ``src`` under ``dst``, refusing to clobber.

    Prefers :func:`os.link`, which publishes ``src``'s own inode under a second
    name. On a filesystem without hard links (FAT/exFAT, some network mounts)
    it copies the bytes into a fresh ``O_CREAT|O_EXCL`` file instead, which is
    still no-clobber but publishes a *different* inode. The mechanism is
    reported rather than left to be inferred from a stat comparison: two
    identities that happen to differ do not prove a copy ran, because ``src``
    can be replaced between this call's stat and its link.

    The source's permission bits are carried over on the copy path; the link
    path preserves them inherently.

    :param src: An existing regular file.
    :param dst: The destination, which must not exist.
    :param fallback_errnos: The set of :func:`os.link` errnos that trigger the
        copy fallback rather than being re-raised. A caller publishing an
        existing user-visible file wants a narrower set than one publishing its
        own temp file: on Linux with ``fs.protected_hardlinks=1``, ``EPERM``
        means "policy denies hardlinking a file you do not own," not "this
        filesystem has no hard links," so treating it as link-less turns a
        denial into copy-then-unlink.
    :returns: The publication mechanism and the identity described above.
    :raises FileExistsError: If ``dst`` exists at publication time, from either
        path.
    :raises OSError: If the link fails for a reason outside the fallback set,
        if the copy-source guards are unavailable on this platform, if the copy
        source is not a regular file at open time, or if the copy itself fails.
        A partial copy is withdrawn before the raise, but only while ``dst``
        still names the inode the exclusive create made.
    """
    # Before publishing, not after: on the link path dst and src name one inode,
    # so a stat taken through either name afterwards records whatever that name
    # holds at that moment — the very thing an identity check must not assume.
    src_stat = os.stat(src)
    try:
        os.link(src, dst)
    except FileExistsError:
        raise
    except OSError as error:
        if error.errno not in fallback_errnos:
            raise
    else:
        return Published("link", src_stat)

    flags = nofollow_read_flags()
    if flags is None:
        raise OSError("cannot copy safely on this platform")
    sfd = os.open(src, flags)
    try:
        if not stat.S_ISREG(os.fstat(sfd).st_mode):
            raise OSError(f"{src} is not a regular file")
        fd = os.open(dst, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        created = os.fstat(fd)
    except BaseException:
        os.close(sfd)
        raise
    # Separate from the arm below: 'created' and 'fd' do not exist yet if we
    # reach that one, and dst must not be withdrawn when we never created it.
    try:
        # The mode comes from src, not the umask: this publishes an existing
        # file, so the copy must carry the permission bits the source held at
        # publication. The 0o600 above is only the create mode, narrowed until
        # fchmod runs and before any byte is written.
        os.fchmod(fd, stat.S_IMODE(src_stat.st_mode))
        with os.fdopen(fd, "wb") as handle:
            fd = -1  # ownership transferred to the file object
            with os.fdopen(sfd, "rb") as source:
                sfd = -1  # ownership transferred to the file object
                while chunk := source.read(io.DEFAULT_BUFFER_SIZE):
                    handle.write(chunk)
            handle.flush()
            # Post-write stat: the same inode as 'created', with the size and
            # mtime the content gave it.
            return Published("copy", os.fstat(handle.fileno()))
    except BaseException:
        if sfd != -1:
            os.close(sfd)
        if fd != -1:
            os.close(fd)
        # Never leave a partial no-clobber target behind: withdraw only while
        # dst still names the inode the exclusive create made.
        try:
            current = dst.lstat()
            if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                dst.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
        raise


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    The content is written to a temporary file in the same directory and then
    moved into place with :func:`os.replace`, so a crash mid-write never leaves
    a partially-written target.

    Identical content is not rewritten (the mtime is preserved) when the target is
    a regular, un-hardlinked file and the exact bytes match. That check needs
    ``O_NOFOLLOW`` and ``O_NONBLOCK`` to be safe, so on a platform missing either
    constant it is not made and every write is a real write.

    Writes are byte-exact: UTF-8, no newline translation.

    :param path: Destination file.
    :param text: Content to write.
    """
    # Skip identical rewrites: generated files keep their mtime, so
    # mtime-based staleness checks (stack status) see no phantom drift
    # after a dry-run re-render. Only a regular, un-hardlinked file may
    # be skipped — symlinks, extra links, and special files must still
    # be replaced — and the comparison is on exact bytes so newline
    # differences count as changes. The skip is bound to one file descriptor
    # and path identity is rechecked so a concurrent swap falls through to a
    # real write. The whole path runs only when nofollow_read_flags() yields flags:
    # without both open flags the open can block on a FIFO or land on a symlink's
    # target, and the recheck comes too late to prevent either. Declining the
    # skip costs a rewrite of unchanged content — a new inode and a fresh
    # mtime, which is exactly the phantom drift described above.
    flags = nofollow_read_flags()
    if flags is not None:
        try:
            fd = os.open(path, flags)
            try:
                st_fd = os.fstat(fd)
                if stat.S_ISREG(st_fd.st_mode) and st_fd.st_nlink == 1:
                    with os.fdopen(fd, "rb") as handle:
                        fd = -1  # ownership transferred to file object
                        current_bytes = handle.read()
                    if current_bytes == text.encode("utf-8"):
                        # Re-verify path identity: same inode as the fd we read from?
                        st_path = os.lstat(path)
                        if (st_fd.st_dev, st_fd.st_ino) == (st_path.st_dev, st_path.st_ino):
                            return  # identical content, verified no swap
            finally:
                if fd != -1:
                    os.close(fd)
        except OSError:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        # mkstemp creates the file 0600; relax it to the conventional file mode
        # (honoring the process umask) so generated config files are readable
        # like the hand-authored sources alongside them.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def atomic_write_new(path: Path, text: str) -> os.stat_result:
    """Write ``text`` to ``path`` atomically, failing if ``path`` exists.

    The content is written to a temporary file and published with
    :func:`link_or_copy_no_replace`, which prefers :func:`os.link` — refusing
    to replace an existing target — and copies the bytes into an exclusive
    create on a filesystem without hard links. ``FileExistsError`` semantics
    are identical on both paths; only the link path publishes the temporary
    file's own inode.

    Writes are byte-exact: UTF-8, no newline translation.

    :param path: Destination file (must not exist).
    :param text: Content to write.
    :returns: The stat of the published inode, captured race-free: the
        temporary file's own stat on the link path, the created inode's on the
        copy path. This function ignores :attr:`Published.kind`, which is sound
        only because its source is a private ``mkstemp`` name no other process
        can replace — a caller publishing a user-visible source must not.
    :raises FileExistsError: If ``path`` already exists at publication time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        return link_or_copy_no_replace(tmp_name, path).stat
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _open_lock_file(path: Path) -> int:
    """Open the lock file at ``path``, creating it and its parent if absent.

    Split out of :func:`name_lock` so a lock found to be on an inode that is no
    longer at ``path`` can be reopened without a second copy of these guards.

    :param path: Lock file.
    :returns: An open descriptor, or ``-1`` when there is no descriptor to be
        had and the caller should degrade to a no-op rather than refuse.
    :raises ConfigError: For the shapes :func:`name_lock` documents as refused.
    :raises OSError: For the shapes :func:`name_lock` documents as raised.
    """
    # 0o666 & ~umask, matching atomic_write_new, so the lock file follows the
    # same convention as everything else stack writes. Both guard flags, since
    # the target's type is not known in advance: O_NOFOLLOW refuses a planted
    # symlink, and O_NONBLOCK keeps a planted FIFO or device node from parking
    # the open before the regular-file check in the caller can refuse it. O_RDWR
    # on a FIFO happens to return immediately on Linux and the BSDs, but POSIX
    # does not define that open at all, and the flag costs nothing on the regular
    # file this is in every real case. Unlike the retry below, this open cannot
    # be declined when a constant is missing — there is no lock without it — so
    # a missing one is 0 and costs that single guard rather than the lock.
    fd = -1
    retried = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR | _O_NOFOLLOW | _O_NONBLOCK, 0o666)
    except (FileExistsError, NotADirectoryError):
        # Something that is not a directory stands where one has to be: at
        # .locks, or — since parents=True walks up — at any ancestor above it.
        # Which of the two errnos mkdir(exist_ok=True) reports depends on how
        # it meets the offender, not on where the offender is. EEXIST when the
        # offender is the final component of a step mkdir attempts: anything at
        # .locks, whatever its type, and also a symlink to nowhere higher up,
        # which mkdir retries as a component to create once the first attempt
        # gives ENOENT. ENOTDIR when the offender is something mkdir has to
        # traverse: any other non-directory above .locks, including a symlink
        # that resolves to one. A symlink above .locks that the kernel will not
        # resolve — a loop, or a chain longer than it will follow, which is a
        # smaller budget than SYMLOOP_MAX suggests and differs for absolute and
        # relative targets — is the single shape neither arm takes: mkdir
        # reports ELOOP, which surfaces as the bare OSError the docstring
        # describes, even where the chain would have resolved to a perfectly
        # good directory. Neither errno is a permission
        # error, so no branch below sees them either. Unlike a permission
        # problem, degrading buys nothing here: no name under this root could
        # ever take a lock, and anyone who can write a shared root can plant
        # one. So refuse, the way a non-regular file at the lock path itself is
        # refused — and name the offender rather than the lock path, because a
        # message pointing at a .locks that does not exist sends the user
        # looking for nothing.
        blocker = next(
            (
                ancestor
                for ancestor in (path.parent, *path.parent.parents)
                if os.path.lexists(ancestor) and not ancestor.is_dir()
            ),
            path.parent,
        )
        raise ConfigError(
            f"Not a directory: {blocker}",
            hint="Move or remove it and retry; stack only ever creates a directory here.",
        ) from None
    except PermissionError:
        # A shared config root whose .locks directory or lock file belongs to
        # another user: under a typical 022 umask that leaves them 0755 and
        # 0644, neither of which this user may write. Read access is enough to
        # lock, since flock takes the open file description and not the access
        # mode, so try that before giving up. Widening the create mode would not
        # help — it cannot reach a file stack neither created nor owns, which
        # is precisely this case. (On a CPython build without HAVE_FLOCK the
        # F_SETLK emulation needs a writable fd and reports EBADF, reaching
        # the same degrade as an unlockable filesystem does in name_lock.)
        #
        # O_NONBLOCK is load-bearing on this open in a way it is not on the
        # create: a read-only open of a FIFO waits for a writer, and nothing
        # here will ever supply one. Without the flag, a FIFO planted by the
        # user who owns the path parks every writer of that name forever —
        # past any timeout, the create having failed EACCES before the
        # regular-file check could refuse it. This open can be declined, unlike
        # the create, so a platform missing either constant — nofollow_read_flags
        # returning None — skips it and takes the degrade below rather than
        # opening a target it cannot guard.
        retry_flags = nofollow_read_flags()
        if retry_flags is not None:
            retried = True
            try:
                fd = os.open(path, retry_flags)
            except OSError as exc:
                if exc.errno not in _LOCK_UNOBTAINABLE_ERRNOS:
                    # Not a permission problem after all: an object was swapped
                    # in at the path between the two opens. Silently degrading
                    # would cost exclusion, so let it surface.
                    raise
    if fd == -1:
        # Neither open got a descriptor, which reads as a permission problem —
        # but the descriptor is the whole mechanism, so whatever the reason,
        # there is no exclusion to be had here. lstat says whether that is
        # because something is sitting at the path or because nothing is; it
        # needs no permission on the file itself, only search on .locks, so it
        # sees an object even at mode 0000.
        try:
            found = os.lstat(path)
        except OSError:
            found = None
        if found is not None and not stat.S_ISREG(found.st_mode):
            raise ConfigError(
                f"Lock file is not a regular file: {path}",
                hint="Remove it and retry; stack only ever creates a plain file here.",
            )
        if found is not None and retried:
            # A regular file that neither open could touch: a lock another user
            # created under a umask that left it unreadable, or one whose mode
            # was changed afterwards. Degrading here would hand back a lock that
            # does not lock, silently, for every process that meets this file —
            # which is the failure this whole context manager exists to prevent,
            # so refuse instead. The fchmod in the caller keeps stack's own locks
            # out of this case whatever umask created them, so a root that only
            # ever sees stack does not reach it.
            raise ConfigError(
                f"Cannot open the lock file: {path}",
                hint=(
                    "Remove it and retry, or make it readable to this user; "
                    "stack cannot serialize this name without opening it."
                ),
            )
        # Nothing at the path, a .locks this user may not even search, or — where
        # the guard flags are missing — a file the read-only retry was declined
        # before it could reach, the one route here that can leave a readable file
        # unread. None of the three is an object this user has been shown to be
        # locked out of, and a .locks that cannot be searched disables locking for
        # every name under it, so refusing one name would not have saved the
        # others. Degrade to the same no-op an unlockable filesystem takes rather
        # than failing a create the rest of stack would complete — os.replace and
        # os.link need the directory the data lives in, not this one. A root that
        # is genuinely unusable fails a moment later, at the write, naming the
        # file the user actually asked for. Reporting the degrade back to the
        # caller rather than yielding inside the handler keeps the caller's own
        # exceptions from being chained to a lock-file error that has nothing to
        # do with them.
        return -1
    return fd


def _swapped_out(fd: int, path: Path) -> bool:
    """Report whether ``path`` has stopped naming the file behind ``fd``.

    :param fd: Descriptor held on the lock file.
    :param path: Lock file the descriptor was opened from.
    :returns: ``True`` if ``path`` names something else, or nothing at all —
        either way the next process opens an object this descriptor does not
        cover. ``False`` when the question cannot be answered.
    """
    held = os.fstat(fd)
    try:
        current = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        # Some other reason the name cannot be examined — a parent that stopped
        # being searchable, a network mount answering ESTALE — which says
        # nothing about identity. Report no swap: this caller holds a lock the
        # kernel granted, and throwing it away over a question that was never
        # answered is not a trade worth making. What the reopen would do
        # instead is not one thing — an unsearchable parent degrades, a
        # non-directory one refuses, a cleared transient succeeds — so the
        # argument for keeping the lock is the one that holds whatever happens:
        # it is at worst the behaviour from before this check existed.
        return False
    return (current.st_dev, current.st_ino) != (held.st_dev, held.st_ino)


@contextmanager
def name_lock(path: Path, name: str, *, timeout: float | None = None) -> Iterator[None]:
    """Hold an exclusive interprocess lock at ``path`` for the duration of the block.

    Serializes whole create-a-name operations across processes, so a pre-check,
    the publish it guards, and the post-check that confirms it cannot be
    interleaved with another process doing the same. The kernel releases the
    lock when the holder exits or dies, so a killed holder never wedges the
    name.

    The lock file is created on first use and **never removed**. Unlinking a
    file another process may already hold open would let a third process create
    a fresh file at the same path and take a lock that excludes nobody, so the
    empty file is left behind deliberately — and so the timeout error below
    points at the process holding the lock rather than at the file.

    Because ``flock`` binds to the open file description and not to the name, a
    lock is handed back only once ``path`` is confirmed to still name the file
    the descriptor was opened on. Without that check, a lock file replaced
    between the open and the grant leaves the two processes holding locks on
    different inodes and both inside the block. On a mismatch this releases and
    reopens against whatever stands there now, sharing the one timeout rather
    than starting a new one. The check covers the grant and nothing past it: a
    holder whose file is replaced while the block runs keeps a lock no later
    process can see, and no check made before the block can tell it so.

    Where ``fcntl`` does not exist, where the filesystem refuses to lock (any
    errno outside the contended set — some NFS and FUSE mounts return ENOLCK
    or EOPNOTSUPP), or where there is no lock file to be had — because this user
    may not write the shared config root, or because reopening it read-only
    would take open guards this platform lacks — this is a no-op that yields
    immediately; what that gives up is stated at each call site. Permissions
    that the rest of ``stack`` writes fine are therefore never turned into a
    root where every create fails.

    What is refused rather than degraded is an object that *is* at ``path`` and
    that neither open could obtain a descriptor for, whatever its type. Without
    a descriptor there is no exclusion, so degrading would hand back a lock that
    does not lock, silently, for as long as that object sits there — and it
    costs nothing to save. Two things hide such an object from that refusal: a
    ``.locks`` this user may not even search, which hides the whole directory
    with it, and a platform without the open guards, where the read-only retry
    is never taken and so a regular file is never shown to be unopenable. To
    keep the strictness off a lock ``stack`` itself created, each acquisition
    re-applies mode ``0o666`` to the file: the create is subject to whichever
    umask reached the name first, and a ``077`` would otherwise leave every
    other user of a shared root locked out of it. That re-apply is best effort
    — a mode change can be refused even to the file's own owner, as macOS does
    for a file flagged ``uchg`` — and where it does not take, a ``0600`` lock
    keeps that mode, and whoever cannot open it then meets the refusal above.

    :param path: Lock file. Its parent directory is created if absent and if
        this user may create it.
    :param name: The profile/bundle/environment name being created, for the
        timeout message.
    :param timeout: Seconds to wait. ``None`` reads the module default at call
        time.
    :raises ConfigError: If the lock is still held when the timeout expires, if the timeout expires
        with ``path`` still being replaced faster than a descriptor can be confirmed on it, if
        something stands at ``path`` that this user cannot obtain a descriptor for and the kernel
        refused only on this user's permissions while still letting them search the parent, if
        something other than a regular file stands there and the kernel let this user open it, or if
        any non-directory — including a symlink to nowhere or to a non-directory — stands at
        ``path``'s parent or at any ancestor of it that would have to be created. The offender is
        named, which is not always the parent. What the kernel declines to open for a reason of its
        own escapes this and raises ``OSError`` below: a symlink at ``path``, a socket whose type it
        will not open, a device node with no driver behind it, a directory, and a symlink above the
        parent it will not resolve — a loop, or a chain past its link budget. A FIFO is the shape
        that never escapes, at any mode. A parent this user may not search hides whatever stands
        below it, so the whole directory degrades rather than refusing or raising. All of that
        assumes both open guards, and so does every promise above it. POSIX requires both, and
        ``fcntl`` — which gates this function entirely — ships only where POSIX does, so the
        guardless arm is defensive rather than reachable; the suite drives it by substituting the
        constants. Which shape it refuses there is deliberately not promised here: the open resolves
        a symlink that the ``lstat`` behind it does not, so the two halves of that arm disagree
        about what is standing at the path, and no single rule covers both.
    :raises OSError: If the lock file cannot be opened for a reason that is
        neither of those and not a permission problem. An over-long name, a
        symlink planted at ``path``, a socket or a device node the kernel will
        not open, a directory at ``path``, and an unresolvable symlink at an
        ancestor of ``path``'s parent are the reachable ones; the list is not
        closed, so a filesystem that fails an open some other way surfaces here
        rather than at the write it guards.
    """
    if not _LOCK_AVAILABLE:
        yield
        return

    import fcntl

    limit = _LOCK_TIMEOUT if timeout is None else timeout
    # One deadline for the whole acquisition, so a lock file being replaced
    # underneath this cannot buy itself a fresh timeout on every reopen.
    deadline = time.monotonic() + limit
    while True:
        fd = _open_lock_file(path)
        if fd == -1:
            yield
            return
        try:
            # flock on a FIFO or device node fails with an errno outside the
            # contended set, which would take the degrade branch below and make
            # this a silent no-op for the name — the one failure this lock must
            # never have. Anyone who can write to a shared config root can plant
            # one, so refuse loudly instead.
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ConfigError(
                    f"Lock file is not a regular file: {path}",
                    hint="Remove it and retry; stack only ever creates a plain file here.",
                )
            # 0o666 flat, not 0o666 & ~umask like everything else stack writes.
            # The create takes whichever umask the first user to reach this name
            # happened to have, and a 077 leaves a lock file no other user can
            # open — which costs every one of them their exclusion, since a
            # descriptor is the whole mechanism. The mode is not for this
            # process, which already holds the fd; it is for the next user, and
            # it is why the refusal in _open_lock_file can be strict without
            # breaking a shared root. The descriptor and not the path, so that an
            # object swapped in after the open is never what gets widened.
            # Failure does not mean the file is someone else's — a mode change
            # can be refused to the owner too, as macOS does for a file flagged
            # uchg — but the descriptor is in hand either way, so raising here
            # would fail this create over a courtesy to the next user. What that
            # costs them is in the docstring.
            with suppress(OSError):
                os.fchmod(fd, 0o666)
            locked = False
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except OSError as exc:
                    if exc.errno not in _LOCK_CONTENDED_ERRNOS:
                        # This filesystem cannot lock. Degrade to the no-op the
                        # missing-fcntl path takes rather than stalling out the
                        # deadline and blaming a competitor that does not exist.
                        break
                    if time.monotonic() >= deadline:
                        raise ConfigError(
                            "Timed out waiting for another stack process to finish "
                            f"creating '{name}'",
                            hint=(
                                "Another stack process may be stuck; retry, or find "
                                f"the process holding {path} (lsof) and stop it. "
                                "Deleting the lock file releases nothing and lets "
                                "the next writer straight past it."
                            ),
                        ) from None
                    time.sleep(_LOCK_POLL)
            if locked and _swapped_out(fd, path):
                # flock binds to the open file description, not to the name, so
                # a lock on an inode no longer at path excludes nobody: the next
                # process opens whatever is there now and locks that instead.
                # Nothing in stack ever removes a lock file, so a mismatch is
                # always an outside actor — but the cost of not looking is two
                # processes inside one critical section, so look, and reopen
                # against the object that is there now. This buys the grant and
                # only the grant: a holder whose file is replaced after this
                # check keeps a lock nobody else can see, and no check made
                # before the block runs can tell it so.
                fcntl.flock(fd, fcntl.LOCK_UN)
                if time.monotonic() >= deadline:
                    raise ConfigError(
                        f"Timed out locking '{name}': {path} keeps being replaced",
                        hint=(
                            "Something outside stack is recreating the lock file; "
                            "stack never removes one. Stop it and retry."
                        ),
                    )
                # Same poll the contended loop uses. Reaching here twice means
                # something is replacing the file faster than it can be locked,
                # and a hot loop on that would burn the whole timeout at full
                # tilt; one poll interval on the ordinary single reopen is not
                # worth avoiding it.
                time.sleep(_LOCK_POLL)
                continue
            try:
                yield
            finally:
                if locked:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            return
        finally:
            os.close(fd)


def probe_locking(path: Path) -> bool:
    """Report whether this filesystem can serve an advisory lock at ``path``.

    Answers the question :func:`name_lock` answers silently, and answers it
    total: **this function never raises.** Contention counts as success,
    because another holder proves the mechanism works. Everything else that
    stops a lock being taken — an absent :mod:`fcntl`, a lock file that cannot
    be obtained, any shape :func:`name_lock` refuses, a ``flock`` declined for
    a reason other than contention — is reported as ``False``. A diagnostic
    that aborts the diagnosis it is part of would be worse than the silence it
    replaces.

    :param path: The probe lock file. Created if absent, and left in place —
        ``flock`` binds to an open file description, so unlinking the name
        would not release anything and would race every other user of the
        directory.
    :returns: ``True`` when a lock could be taken or was held by someone else,
        ``False`` for every other outcome.
    """
    if not _LOCK_AVAILABLE:
        return False

    import fcntl

    fd = -1
    try:
        fd = _open_lock_file(path)
        if fd == -1:
            return False
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            # name_lock raises here, because a planted FIFO costs it its
            # exclusion. The probe reports instead: "locking is degraded" is
            # the honest answer either way, and a diagnostic must not abort.
            return False
        # The same courtesy name_lock pays the next user, for the same reason:
        # probe.lock is persistent, so a first run under a 077 umask would
        # otherwise leave a file no other user of a shared root can open —
        # making the probe the cause of the degrade it reports.
        with suppress(OSError):
            os.fchmod(fd, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            return exc.errno in _LOCK_CONTENDED_ERRNOS
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except (ConfigError, OSError):
        # Load-bearing, not defensive. _open_lock_file raises ConfigError for a
        # non-directory at .locks or any ancestor, for a non-regular file at the
        # lock path, and for a regular lock file neither open could touch; and a
        # bare OSError for the shapes the kernel declines itself. Every one is a
        # root on which name_lock cannot serialize anything, which is precisely
        # what the caller asked about.
        return False
    finally:
        if fd != -1:
            # Suppressed, not ignored: close can fail (EIO on a flaky network
            # mount is the realistic one), and an exception from a finally
            # replaces whatever the body returned — which would break the
            # never-raises contract on precisely the degraded roots this
            # function exists to report.
            with suppress(OSError):
                os.close(fd)
