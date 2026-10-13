"""Modal SDK adapter: CPU and GPU sandboxes with hard CPU/RAM limits and a provider-side lifetime.

Modal bills a GPU sandbox's cores and memory on top of its GPU, which a static GPU price never sees. GPU
sandboxes therefore default to 8 cores and 64 GiB per GPU and are capped at 16 cores and 128 GiB per GPU.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from decimal import Decimal
import hashlib
from pathlib import Path
import re
import tempfile

from ..pricing import modal_gpu_model
from . import build, files
from .base import (OwnedProvider, ProviderError, ProviderRejectedError, ProviderUncertainError, ResourceGoneError,
                   lease_name, quote)

try:  # The SDK is optional: tests use fakes, and agent sandboxes never import the server.
    from modal.exception import ConflictError, InvalidError
except ImportError:
    ConflictError = InvalidError = ()

# Registry references such as "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime".
IMAGE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,254}")
MAX_GPUS = 8
CPUS_PER_GPU, MEMORY_GIB_PER_GPU = 8, 64
MAX_CPUS_PER_GPU, MAX_MEMORY_GIB_PER_GPU = 16, 128
MAX_CPUS, MAX_MEMORY_GIB = 256, 2048  # CPU-only sandboxes


class ModalProvider(OwnedProvider):
    provider = "modal"
    # poll() reports an exit code only once the sandbox has really ended, so the broker may persist "stopped".
    status_is_final = True
    # The lease id travels in the sandbox's tags, so a lookup that finds nothing (after the reaper's
    # grace period) proves the create never took effect and its hold can be released.
    absence_is_final = True

    def __init__(self, sdk, *, app, rates, image=None, clock=None):
        """rates: {"gpu_hour_usd": {Modal GPU name: rate}, "cpu_core_hour_usd", "memory_gib_hour_usd", "version"}."""
        super().__init__(**({"clock": clock} if clock else {}))
        # image is the default for leases that don't name one; None uses Modal's Debian slim.
        self.sdk, self.app, self.image, self.rates = sdk, app, image, rates
        self.handles, self.images = {}, {}

    def offers(self):
        cpu = [(f"cpu-{n}", {"gpus": 0, "cpus": n, "memory_gib": 2 * n, "max_seconds": 600}) for n in (1, 2, 4, 8, 16, 32)]
        gpu = [(name, {"gpu": name, "gpus": 1, "cpus": CPUS_PER_GPU, "memory_gib": MEMORY_GIB_PER_GPU, "max_seconds": 600})
               for name in self.rates["gpu_hour_usd"]]
        result = []
        for offer_id, spec in cpu + gpu:
            name = spec.get("gpu")
            limits = ({"max_seconds": 86400, "gpus": MAX_GPUS, "cpus_per_gpu": MAX_CPUS_PER_GPU,
                       "memory_gib_per_gpu": MAX_MEMORY_GIB_PER_GPU} if name else
                      {"max_seconds": 86400, "gpus": 0, "cpus": MAX_CPUS, "memory_gib": MAX_MEMORY_GIB})
            rates = {"cpu_core_hour_usd": self.rates["cpu_core_hour_usd"], "memory_gib_hour_usd": self.rates["memory_gib_hour_usd"],
                     **({"gpu_hour_usd": self.rates["gpu_hour_usd"][name]} if name else {})}
            result.append({"id": offer_id, "provider": self.provider, "gpu": name,
                           **({"gpu_model": modal_gpu_model(name)} if name else {}), "gpus": spec["gpus"],
                           "cpus": spec["cpus"], "memory_gib": spec["memory_gib"], "hourly_usd": self.quote(spec)["hourly_usd"],
                           "spec": spec, "limits": limits, "rates": rates, "rate_version": self.rates.get("version"),
                           "custom_image": True, "dockerfile": True, "network_off": True})
        return result

    def quote(self, spec):
        allowed = {"gpu", "gpus", "cpus", "memory_gib", "max_seconds", "image"} | build.FIELDS
        if set(spec) - allowed:
            raise ValueError("Unsupported Modal specification fields")
        gpu = spec.get("gpu")
        gpus = spec.get("gpus", 0 if gpu is None else 1)
        if type(gpus) is not int or not 0 <= gpus <= MAX_GPUS:
            raise ValueError(f"gpus must be an integer in [0, {MAX_GPUS}]")
        if gpus:
            if not isinstance(gpu, str) or gpu not in self.rates["gpu_hour_usd"]:
                raise ValueError("Unknown Modal GPU; choices: " + ", ".join(self.rates["gpu_hour_usd"]))
            cpus, memory = spec.get("cpus", CPUS_PER_GPU * gpus), spec.get("memory_gib", MEMORY_GIB_PER_GPU * gpus)
            limits = {"cpus": MAX_CPUS_PER_GPU * gpus, "memory_gib": MAX_MEMORY_GIB_PER_GPU * gpus}
        elif gpu is not None:
            raise ValueError("CPU-only resources (gpus 0) must omit gpu or set it to null")
        else:
            cpus, memory = spec.get("cpus", 8), spec.get("memory_gib", 32)
            limits = {"cpus": MAX_CPUS, "memory_gib": MAX_MEMORY_GIB}
        for key, value in (("cpus", cpus), ("memory_gib", memory)):
            if type(value) is not int or not 1 <= value <= limits[key]:
                per_gpu = f" ({limits[key] // gpus} per GPU)" if gpus else ""
                raise ValueError(f"{key} must be an integer in [1, {limits[key]}]{per_gpu}")
        rate = (Decimal(self.rates["cpu_core_hour_usd"]) * cpus + Decimal(self.rates["memory_gib_hour_usd"]) * memory
                + (Decimal(self.rates["gpu_hour_usd"][gpu]) * gpus if gpus else 0))
        image = spec.get("image")
        if image is not None and not (isinstance(image, str) and IMAGE_REF.fullmatch(image)):
            raise ValueError("image must be a container registry reference, e.g. pytorch/pytorch:latest")
        extra = build.validate(spec)
        q = quote(self.provider, {"gpu": gpu if gpus else None, "gpus": gpus, "cpus": cpus, "memory_gib": memory,
                  "max_seconds": spec.get("max_seconds"), **({"image": image} if image else {}), **extra}, rate,
                  "provider_max_lifetime")
        # The machine as static prices see it: Modal's H100 is an SXM part.
        q.update(gpu=modal_gpu_model(gpu) if gpus else None, gpus=gpus, cpus=cpus)
        return q

    def _dockerfile_image(self, dockerfile, context):
        raw = build.context_tar(context) if context else b""
        key = "dockerfile:" + hashlib.sha256(dockerfile.encode() + b"\0" + raw).hexdigest()
        if key not in self.images:
            with tempfile.TemporaryDirectory(prefix="spin-build-") as directory:
                root = Path(directory)
                if raw:
                    build.extract(raw, root / "context")
                (root / "context").mkdir(exist_ok=True)
                (root / "Dockerfile").write_text(dockerfile)
                image = self.sdk.Image.from_dockerfile(root / "Dockerfile", context_dir=root / "context")
                try:
                    # Build while the context exists; a failed build creates no sandbox and costs nothing.
                    self.images[key] = image.build(self.app)
                except Exception as exc:
                    raise ProviderRejectedError("Modal could not build the Dockerfile: "
                                                + (str(exc).splitlines() or [type(exc).__name__])[0][:300]) from exc
        return self.images[key]

    def _image(self, ref):
        if not ref and self.image:
            return self.image  # The operator's prebuilt image.
        if ref not in self.images:
            image = self.sdk.Image.from_registry(ref) if ref else self.sdk.Image.debian_slim(python_version="3.12")
            try:
                # Pull before creating a sandbox so a bad reference fails cleanly and free of charge.
                self.images[ref] = image.build(self.app)
            except Exception as exc:
                raise ProviderRejectedError(f"Modal could not load image {ref or 'default'}: "
                                            + (str(exc).splitlines() or [type(exc).__name__])[0][:200]) from exc
        return self.images[ref]

    def create(self, spec, *, lease_id):
        q = self.quote(spec)
        s = q["spec"]
        image = (self._dockerfile_image(spec["dockerfile"], spec.get("context")) if spec.get("dockerfile")
                 else self._image(s.get("image")))
        name, started = lease_name(lease_id), self.clock()
        hardware = {"gpu": f'{s["gpu"]}:{s["gpus"]}'} if s["gpus"] else {}
        try:
            sb = self.sdk.Sandbox.create("sleep", str(s["max_seconds"]), app=self.app,
                name=name, tags={"spin_lease": lease_id}, image=image, timeout=s["max_seconds"], **hardware,
                cpu=(s["cpus"], s["cpus"]), memory=(s["memory_gib"] * 1024, s["memory_gib"] * 1024),
                secrets=[], include_oidc_identity_token=False, block_network=s.get("network") == "none")
            self.handles[sb.object_id] = sb
            return self._created(sb.object_id, lease_id, q, started, name=name, hourly_usd=q["hourly_usd"])
        except Exception as exc:
            raise ProviderUncertainError(f"Modal create uncertain; reconcile name {name}") from exc

    def recover(self, *, lease_id, spec, created_at, maximum_usd):
        """Find a sandbox whose create response was lost, by its lease tag; never create another."""
        found = list(self.sdk.Sandbox.list(app_id=getattr(self.app, "app_id", None), tags={"spin_lease": lease_id}))
        if len(found) > 1:
            raise ProviderError("Ambiguous Modal lease lookup; operator reconciliation required")
        if not found:
            return None
        sandbox = found[0]
        self.handles[sandbox.object_id] = sandbox
        q = {"max_seconds": spec["max_seconds"], "max_cost_usd": maximum_usd}
        gpus = spec.get("gpus", 0 if spec.get("gpu") is None else 1)
        record = self._created(sandbox.object_id, lease_id, q, created_at, name=lease_name(lease_id),
                               hourly_usd=str(Decimal(maximum_usd) * 3600 / spec["max_seconds"]), recovered=True,
                               gpu=modal_gpu_model(spec["gpu"]) if gpus else None, gpus=gpus)
        code = sandbox.poll()
        if code is not None:  # Exited already; Modal billed it, so the lease is charged and recorded stopped.
            self.records[sandbox.object_id].update(status="stopped", exit_code=code)
            record.update(status="stopped", exit_code=code)
        return record

    def _handle(self, resource_id):
        self._record(resource_id)
        if resource_id not in self.handles:
            self.handles[resource_id] = self.sdk.Sandbox.from_id(resource_id)
        return self.handles[resource_id]

    def _exited(self, resource_id):
        """After an SDK failure: the ResourceGoneError to raise if the sandbox has already ended, else None."""
        handle = self.handles.get(resource_id)
        try:
            code = handle.poll() if handle is not None else None
        except Exception:
            code = None  # Unknown; the caller reports the original failure.
        if code is None:
            return None
        self._record(resource_id).update(status="stopped", exit_code=code)
        return ResourceGoneError(f"Modal sandbox has exited (exit code {code}); create a new resource")

    @contextmanager
    def _sdk(self, resource_id):
        """SDK failures become broker errors: exited sandbox 409, refused command 400, anything else 502."""
        try:
            yield
        except (ProviderError, ValueError):
            raise  # Already classified, by this adapter or by the stream's consumer.
        except Exception as exc:
            gone = self._exited(resource_id)
            if gone:
                raise gone from exc
            if isinstance(exc, InvalidError) and not isinstance(exc, ConflictError):
                # Client-side validation; its text is replaced because it may echo the command.
                raise ValueError("Modal rejected the command (for example arguments over 64 KiB); "
                                 "use spin compute upload for large payloads") from exc
            raise ProviderError(f"Modal SDK failure ({type(exc).__name__})") from exc

    def exec(self, resource_id, command, timeout=60):
        timeout = self._timeout(resource_id, timeout)
        with self._sdk(resource_id):
            p = self._handle(resource_id).exec("bash", "-lc", command, timeout=timeout)
            # Drain both pipes simultaneously to avoid stderr filling and deadlocking.
            with ThreadPoolExecutor(max_workers=2) as pool:
                out = pool.submit(p.stdout.read)
                err = pool.submit(p.stderr.read)
                p.wait()
            return {"exit_code": p.returncode, "stdout": out.result(), "stderr": err.result()}

    def status(self, resource_id):
        record = self._record(resource_id)
        with self._sdk(resource_id):
            code = self._handle(resource_id).poll()
        record["status"] = "running" if code is None else "stopped"
        return {**record, "exit_code": code}

    def stop(self, resource_id):
        record = self._record(resource_id)
        if "settled_cost_usd" in record:
            return dict(record)
        try:
            handle = self._handle(resource_id)
            if handle.poll() is None:  # Already exited (e.g. its timeout) needs no terminate.
                handle.terminate(wait=True)
                if handle.poll() is None:
                    raise RuntimeError("Provider has not confirmed stop")
        except Exception as exc:
            # Past its deadline, Modal's own sandbox timeout has ended it; nothing is left to stop.
            if self.clock() < record["expires_at"] + 300:
                raise ProviderUncertainError("Modal stop uncertain; retain reservation") from exc
        return self._stopped(record, "stopped")

    def ssh(self, resource_id):
        self._record(resource_id, active=True)
        return {"supported": False, "reason": "Modal exposes SDK exec, not native SSH; use spin compute exec"}

    # Streams go through exec (`cat`), which works with networking off and keeps whole
    # files out of server memory and disk; Modal's filesystem API only copies whole files.
    @contextmanager
    def read_stream(self, resource_id, path):
        self._record(resource_id, active=True)
        with self._sdk(resource_id):
            sandbox = self._handle(resource_id)
            try:
                sandbox.filesystem.stat(path)
            except Exception as exc:
                gone = self._exited(resource_id)
                if gone:
                    raise gone from exc
                raise ValueError(f"No file at {path}") from None
            process = sandbox.exec("cat", "--", path, text=False)
            yield files.Chunks(process.stdout)
            if process.wait() != 0:
                raise ProviderError(f"Reading {path} failed")

    def write_stream(self, resource_id, path, stream):
        self._record(resource_id, active=True)
        with self._sdk(resource_id):
            sandbox = self._handle(resource_id)
            parent = path.rsplit("/", 1)[0]
            if parent:
                sandbox.filesystem.make_directory(parent, create_parents=True)
            process = sandbox.exec("sh", "-c", 'cat > "$1"', "sh", path, text=False)
            while chunk := stream.read(files.CHUNK):
                process.stdin.write(chunk)
                process.stdin.drain()
            process.stdin.write_eof()
            process.stdin.drain()
            if process.wait() != 0:
                raise ProviderError(f"Writing {path} failed: " + process.stderr.read()[:200].decode(errors="replace"))
        return {"path": path}
