# BuildKit named contexts must be independently pinned/verified Lean and Mathlib trees.
# See README.md. No candidate directory is a build context.
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgmp10 git && rm -rf /var/lib/apt/lists/*
COPY --from=lean_toolchain / /opt/lean/
COPY --from=mathlib / /opt/mathlib/
COPY checker/ProofCheck.lean checker/worker.py simplification_metric.py /checker/
RUN test "$(git -c safe.directory=/opt/mathlib -C /opt/mathlib rev-parse HEAD)" = '9837ca9d65d9de6fad1ef4381750ca688774e608' && \
    test "$(cat /opt/mathlib/lean-toolchain)" = 'leanprover/lean4:v4.15.0' && \
    /opt/lean/bin/lean --version | python -c 'import sys; assert "version 4.15.0" in sys.stdin.read()'
RUN python -c 'import pathlib,subprocess,os; p=pathlib.Path("/opt/mathlib"); libs=[p/".lake/build/lib",*sorted((p/".lake/packages").glob("*/.lake/build/lib"))]; subprocess.run(["/opt/lean/bin/lean","-o","ProofCheck.olean","ProofCheck.lean"], cwd="/checker",env=dict(os.environ,LEAN_PATH=":".join(map(str,libs))),check=True)'
LABEL frontier.lean="4.15.0" frontier.mathlib="9837ca9d65d9de6fad1ef4381750ca688774e608" frontier.protocol="lean-simplification-v1"
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "/checker/worker.py"]
