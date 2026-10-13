"""Modal grading sandbox without block_network: an empty outbound allowlist, plus one encrypted port for the relay.

Modal refuses open ports on a block_network sandbox, so this environment asks Modal for "allowlist" networking with
nothing allowed: every outbound connection is dropped by Modal (silently: a connect hangs until it times out), but its
DNS resolver (172.21.0.1) still answers, and an encrypted tunnel can reach the relay's port. No Modal option turns
that resolver off short of block_network. Inside, netguard.py disables lookups (nsswitch) and makes outbound sockets
fail fast in userspace, and the relay's monitor records (never fails on) any non-local peer the candidate's UID still
reaches for.

It extends RecoverableModalEnvironment (recoverable exec output). `_create_sandbox` mirrors Harbor 0.22.0's
ModalEnvironment._create_sandbox with two changes: no-network becomes empty
outbound allowlists, and the sandbox opens `encrypted_ports=[relay_port]`. Harbor is pinned; a version change must
re-check this copy (tests/test_grading_relay.py pins the parent's source).
"""
from __future__ import annotations

from typing import Any

from harbor.environments import modal as harbor_modal
from tenacity import retry, stop_after_attempt, wait_exponential

from .modal_environment import RecoverableModalEnvironment

CANDIDATE_UID = 65534


class GradingModalEnvironment(RecoverableModalEnvironment):
    network_isolation = "Modal egress allowlist (empty) + userspace guard (no DNS, outbound refused) + monitor"
    monitor_uid = CANDIDATE_UID

    def __init__(self, *args, relay_port=None, **kwargs):
        if type(relay_port) is not int or not 1024 <= relay_port <= 65535:
            raise ValueError("GradingModalEnvironment needs relay_port (1024-65535)")
        self.relay_port = relay_port
        super().__init__(*args, **kwargs)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    async def _create_sandbox(self, *, entrypoint=None, block_network=None, experimental_options=None):
        kwargs: dict[str, Any] = {}
        if experimental_options:
            kwargs["experimental_options"] = experimental_options
        if (cpu := self._cpu_config()) is not None:
            kwargs["cpu"] = cpu
        if (memory := self._memory_config()) is not None:
            kwargs["memory"] = memory
        if (gpu := self._gpu_config()) is not None:
            kwargs["gpu"] = gpu
        if self._network_disabled:
            # What Harbor does for no-network under a dynamic policy: Modal's allowlist mode with nothing allowed.
            kwargs.update(self._dynamic_network_kwargs(self.network_policy))
        elif self._network_is_allowlist:
            kwargs.update(self._allowlist_network_kwargs(self.network_policy))
        if labels := self._sandbox_labels():
            if not self._sandbox_v2_enabled:
                kwargs["tags"] = labels
        if region := self._kwargs.get("region"):
            kwargs["region"] = region
        create = (harbor_modal.Sandbox._experimental_create.aio if self._sandbox_v2_enabled
                  else harbor_modal.Sandbox.create.aio)
        return await create(
            *(entrypoint or ()),
            app=self._app,
            image=self._image,
            timeout=self._sandbox_timeout,
            idle_timeout=self._sandbox_idle_timeout,
            name=self.session_id,
            block_network=False,
            encrypted_ports=[self.relay_port],
            secrets=self._secrets_config(),
            volumes=self._volumes_config(),
            **kwargs,
        )

    async def expose_port(self, port, auth="bearer"):
        """(HTTPS tunnel URL, headers) for the relay's port. The URL is unguessable; the relay checks its own token."""
        if port != self.relay_port:
            raise ValueError(f"only the relay port {self.relay_port} was opened at sandbox creation")
        tunnels = await self._sandbox.tunnels.aio(timeout=120)
        return tunnels[port].url.rstrip("/"), {}
