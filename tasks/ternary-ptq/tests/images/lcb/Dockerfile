FROM python:3.12.11-slim-bookworm
# Build once, publish to a container registry, and configure its immutable image reference. This image
# contains no credentials, datasets, answers, or verifier scoring code. The trusted
# supervisor drops candidate code to a fresh unprivileged UID/GID and enforces time/resource caps.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTIER_LCB_REMOTE=1
RUN mkdir -p /tmp/candidate && chown 65534:65534 /tmp/candidate && chmod 700 /tmp/candidate
USER root
CMD ["sleep", "infinity"]
