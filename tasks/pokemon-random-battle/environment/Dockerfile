FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends bash curl git openssh-client procps rsync rclone ripgrep ca-certificates && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends nodejs && rm -rf /var/lib/apt/lists/*
# Modal appends /root/ to every sandbox's PYTHONPATH; pip fails scanning it unless the actor can list it.
RUN chmod 755 /root
COPY package/ /opt/frontier-ml/
# The `spin` and `frontier` commands.
RUN pip install --no-cache-dir /opt/frontier-ml && spin --version
RUN mkdir -p /task/public
# The public validation script: the same request template and metric as the trusted client.
COPY validation/ /task/
COPY setup.json /task/setup.json
RUN useradd -m -u 1000 actor && mkdir -p /workspace/output /workspace/artifacts /logs/agent && ln -s /workspace/artifacts /artifacts && chown -R actor:actor /workspace
WORKDIR /workspace
