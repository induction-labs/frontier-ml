# Isolated Lean checker

## Official grading: local to the CPU verifier

`task.yaml` selects `checker/verifier.Dockerfile`, which the task builder includes
in the trusted verifier image. The image downloads Lean 4.15.0 with a checked
SHA-256, checks out the pinned Mathlib commit, compiles `ProofCheck.lean`, and
records a build fingerprint. No checker endpoint, secret, separate deployment or
Docker daemon is needed at grading time.

The verifier has 8 CPUs and 40 GiB RAM. `simplification_local.py` starts up to eight
fresh Lean subprocesses using `sandbox.c`: separate unprivileged UIDs, a minimal
read-only chroot, one CPU and 4 GiB private memory per process, 60 seconds wall time, and a syscall
allowlist that blocks networking, child processes, mounts and tracing. Only the
current theorem/proof and trusted toolchain are readable. Compiler output goes to
a bounded file opened by the trusted parent outside the chroot. The isolation
probe and positive/negative proof checks fail preflight if any protection is absent.
Mathlib's read-only mappings share a toolchain cache outside the per-proof private
memory allowance. Address space is capped at 32 GiB. Writable allocations have a
4 GiB `RLIMIT_DATA`, Lean allocation guard, and a trusted parent watchdog that
checks `VmData` every 50 ms and kills over-limit processes. The watchdog is needed
on Modal, which currently reports `RLIMIT_DATA` without enforcing it. Preflight
reserves an oversized writable mapping without touching it and verifies that
either the kernel rejects it or the watchdog kills the probe. The chroot has only
an entropy device and synthetic executable-location links, never a real `/proc`
mount or grader environment.

Generation uses batches of eight with 60 seconds per batch (45 batches for 354
proofs). The H100 is released before correctness checking on the CPU verifier.
Both stages allow about 45 minutes of proof work; the overall Harbor verifier
deadline is two hours, including a source-proof preflight capped at fifteen minutes.
These are operator-selected limits, not claims of difficulty calibration.

From `tasks/lean-simplification`, build the local checker without provisioning a GPU:

```bash
docker build -f checker/verifier.Dockerfile -t frontier-lean-verifier-base:local .
```

`tests/test_lean_simplification.py::test_real_local_lean_checker` exercises the real
backend when `LEAN_SIMPLIFICATION_LOCAL_CHECKER=1` inside a built verifier image.

## Optional public-development HTTP helper

The public validator can also call this HTTP service. Each request starts a fresh
non-root Docker container, offline and read-only, with 1 CPU, 4 GiB RAM, 128 PIDs,
a 32 MiB scratch filesystem and no host mounts. Do not run the service in the
candidate sandbox. Keep it behind an operator-controlled HTTPS endpoint/firewall;
the local default listens on loopback. Never give the candidate its production URL.

Prepare Lean v4.15.0 and Mathlib commit
`9837ca9d65d9de6fad1ef4381750ca688774e608`, including cached dependencies. The
Mathlib checkout must be clean (`git diff --exit-code HEAD`) and include its `.git`
metadata. Build from those trusted trees, with BuildKit named contexts. In a checkout, first enter
`tasks/lean-simplification`; for the shipped reference helper, enter
`/artifacts/lean-checker` and set `PYTHONPATH=/task`:

```bash
docker build \
  --build-context lean_toolchain=/absolute/path/to/lean-v4.15.0 \
  --build-context mathlib=/absolute/path/to/mathlib4 \
  -f checker/Dockerfile \
  -t frontier-lean-simplification:local .
docker image inspect frontier-lean-simplification:local --format '{{.Id}}'
# Supply that exact sha256 image ID, not the mutable tag:
python simplification_checker.py --image sha256:IMAGE_ID
```

The service retains a local fingerprint tag so rebuilding a development tag cannot
remove the image used by running checks.

Pass the startup JSON's fingerprint and URL to public validation using
`--checker-fingerprint` and `--checker-url`. Pin the built image and
its toolchain (the mutable base-image tag is a build recipe, not the runtime pin).
The image ID covers the compiled Lean checker, worker, metric/policy and toolchain.
The service checks positive and negative proofs before serving `/health`.

`ProofCheck.lean` parses the proof as a whole term and inserts its syntax tree into
a trusted declaration. It separately elaborates the organizer's signature, saves
the original environment, and sends the new declarations through
`Lean.Environment.replay` into that original environment before comparing types
and collecting axioms. The input proof is never installed in this environment.
The source policy restricts arbitrary metaprogramming; Docker supplies process
isolation as a separate protection. Candidate output is never executed in the
HTTP service or verifier process. Compiler logs and exit status are parsed by a
trusted worker outside the Lean process; no proof-supplied verdict is accepted.

From the repository root, use
`uv run python tasks/lean-simplification/pipeline/build_data.py --help`
for data preparation, or validate a server with:

```bash
uv run python tasks/lean-simplification/validate.py \
  --data data/lean-simplification/public/validation.jsonl \
  --server-url http://127.0.0.1:8000 --checker-url http://127.0.0.1:8124
```

Final grading always uses the image's local checker. Missing toolchain files,
failed isolation checks, wrong sample counts or invalid source proofs fail
preflight. Infrastructure errors are never converted into invalid-proof labels.
