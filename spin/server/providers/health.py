"""Trusted initial probe: no torch dependency or agent-supplied command.

The broker runs it through the adapter's own exec before the agent gets access, so the machine
must answer commands and, on a GPU lease, every visible GPU must work. The GPU check degrades
with the image: the Python/CUDA-driver probe where python3 and libcuda.so.1 exist, otherwise an
nvidia-smi listing. A GPU machine with neither fails. CPU leases only need to answer.
"""

MARKER = "spin-health-ok"
PROBE_TIMEOUT = 15  # seconds per attempt; cuInit on a cold GPU can take several

CUDA_PROBE = """python3 -I - <<'SPIN_GPU_HEALTH'
import ctypes

cuda = ctypes.CDLL('libcuda.so.1')
def check(name, *args):
    status = getattr(cuda, name)(*args)
    if status:
        raise RuntimeError('%s returned CUDA error %d' % (name, status))

check('cuInit', 0)
count = ctypes.c_int()
check('cuDeviceGetCount', ctypes.byref(count))
if count.value < 1:
    raise RuntimeError('No CUDA devices')
for ordinal in range(count.value):
    context = ctypes.c_void_p()
    check('cuDevicePrimaryCtxRetain', ctypes.byref(context), ordinal)
    try:
        check('cuCtxSetCurrent', context)
        memory = ctypes.c_uint64()
        check('cuMemAlloc_v2', ctypes.byref(memory), ctypes.c_size_t(1048576))
        try:
            check('cuMemsetD8_v2', memory, 0, ctypes.c_size_t(1048576))
            check('cuCtxSynchronize')
        finally:
            check('cuMemFree_v2', memory)
    finally:
        check('cuDevicePrimaryCtxRelease', ordinal)
print('""" + MARKER + """')
SPIN_GPU_HEALTH"""

# Plain POSIX sh: adapters run commands through SSH login shells, bash -lc, or a provider exec API.
GPU_HEALTH_COMMAND = f"""if command -v python3 >/dev/null 2>&1 && python3 -c 'import ctypes; ctypes.CDLL("libcuda.so.1")' >/dev/null 2>&1; then
{CUDA_PROBE}
elif command -v nvidia-smi >/dev/null 2>&1; then
nvidia-smi -L 2>/dev/null | grep -q '^GPU ' && echo {MARKER}
else
echo 'spin-health: no python3 with libcuda.so.1 and no nvidia-smi on this GPU machine' >&2
exit 1
fi"""

REACHABILITY_COMMAND = f"echo {MARKER}"


def probe_command(gpus):
    """The probe for a lease with `gpus` GPUs; unknown hardware (None) gets the GPU probe."""
    return REACHABILITY_COMMAND if gpus == 0 else GPU_HEALTH_COMMAND


def passed(result):
    """An exec result passes only when the probe's marker is its last output line; login banners before it are fine."""
    if not isinstance(result, dict) or result.get("exit_code") != 0:
        return False
    return str(result.get("stdout") or "").strip().splitlines()[-1:] == [MARKER]
