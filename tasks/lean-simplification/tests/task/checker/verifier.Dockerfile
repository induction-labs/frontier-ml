FROM python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git zstd gcc libc6-dev libseccomp-dev libgmp10 && rm -rf /var/lib/apt/lists/*
RUN curl -fL --retry 3 https://github.com/leanprover/lean4/releases/download/v4.15.0/lean-4.15.0-linux.tar.zst -o /tmp/lean.tar.zst && \
    echo 'af71a2569a9f68337de2434829b3008cd8e32c436e9cb6bd8c84a2c2ba3585c9  /tmp/lean.tar.zst' | sha256sum -c - && \
    mkdir -p /opt/lean && tar --zstd -xf /tmp/lean.tar.zst --strip-components=1 -C /opt/lean && rm /tmp/lean.tar.zst
ENV PATH="/opt/lean/bin:$PATH"
RUN git init /opt/mathlib && git -C /opt/mathlib remote add origin https://github.com/leanprover-community/mathlib4.git && \
    git -C /opt/mathlib fetch --depth 1 origin 9837ca9d65d9de6fad1ef4381750ca688774e608 && \
    git -C /opt/mathlib checkout --detach FETCH_HEAD && \
    test "$(cat /opt/mathlib/lean-toolchain)" = 'leanprover/lean4:v4.15.0' && \
    cd /opt/mathlib && lake exe cache get && git diff --exit-code HEAD
COPY checker/ProofCheck.lean /checker/ProofCheck.lean
COPY checker/sandbox.c checker/isolation_probe.c checker/build_local.py /opt/frontier-lean/
RUN gcc -O2 -Wall -Wextra -Werror /opt/frontier-lean/sandbox.c -lseccomp -o /opt/frontier-lean/sandbox && \
    gcc -O2 -Wall -Wextra -Werror /opt/frontier-lean/isolation_probe.c -pthread -o /checker/isolation-probe && \
    python /opt/frontier-lean/build_local.py
ENV PATH="/usr/local/bin:/usr/bin:/bin"
WORKDIR /tmp
