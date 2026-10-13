"""Harbor's pinned Modal backend with recoverable, single-execution output.

Modal can lose an exec's stdio stream after the command has already completed.
Keep a private copy of the output so that recovering it never replays a command
(in particular, an agent session or an inference request).
"""
import asyncio
import uuid

from harbor.environments.base import ExecResult
from harbor.environments.modal import ModalEnvironment, _cached_env_secret


async def _control_call(call, *, attempts=3, timeout=None):
    """Retry only reads, waits on an existing process, and idempotent removal."""
    for attempt in range(attempts):
        try:
            if timeout is None:
                return await call()
            return await asyncio.wait_for(call(), timeout)
        except Exception:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(0.25 * (attempt + 1))


class RecoverableModalEnvironment(ModalEnvironment):
    async def _sdk_exec(self, command, cwd=None, env=None, timeout_sec=None, shell="bash", login=False):
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        env = self._merge_env(env)
        directory = "/tmp/frontier-exec-" + uuid.uuid4().hex
        # The outer shell is the trusted transport's user; Harbor has already
        # wrapped the inner command in `su` when a different user was requested.
        # A subshell preserves exit/exec/pipefail behavior without letting the
        # command overwrite the wrapper's status variable. This also works in sh.
        wrapped = (
            f"mkdir -m 700 {directory} || exit 125\n"
            f"(\n{command}\n) >{directory}/stdout 2>{directory}/stderr\n"
            "_frontier_status=$?\n"
            f"cat {directory}/stdout || exit 125\n"
            f"cat {directory}/stderr >&2 || exit 125\n"
            'exit "$_frontier_status"\n'
        )
        # Never retry creation: an ambiguous error here may mean it did start.
        process = await self._sandbox.exec.aio(
            shell, "-lc" if login else "-c", wrapped, workdir=cwd,
            secrets=[_cached_env_secret(env)] if env else [], timeout=timeout_sec,
        )
        completed = False
        try:
            stream_error = None
            try:
                stdout = await process.stdout.read.aio()
                stderr = await process.stderr.read.aio()
            except Exception as exc:
                stream_error = exc
            # The provider's actual exit status remains authoritative, including
            # -1 for a timeout. Saved output alone never proves success.
            return_code = await _control_call(process.wait.aio)
            completed = True
            if stream_error is not None:
                self.logger.warning("Modal exec stdio failed (%s); reading saved output from the same command",
                                    type(stream_error).__name__)
                filesystem = self._sandbox.filesystem
                stdout = await _control_call(lambda: filesystem.read_text.aio(directory + "/stdout"), timeout=30)
                stderr = await _control_call(lambda: filesystem.read_text.aio(directory + "/stderr"), timeout=30)
            return ExecResult(stdout=stdout, stderr=stderr, return_code=return_code)
        finally:
            # On cancellation the command may still be writing. Its sandbox's
            # normal teardown owns those files; don't interfere with execution.
            if completed:
                try:
                    await _control_call(lambda: self._sandbox.filesystem.remove.aio(directory, recursive=True),
                                        attempts=2, timeout=3)
                except Exception as exc:
                    self.logger.warning("Modal exec output cleanup failed (%s); deferred to sandbox teardown",
                                        type(exc).__name__)
