FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/frontier-rustup CARGO_HOME=/opt/frontier-cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.89.0
ENV PATH=/opt/frontier-cargo/bin:$PATH
COPY foul-play-requirements.txt setup_foul_play.sh /tmp/foul-play-setup/
RUN sh /tmp/foul-play-setup/setup_foul_play.sh /opt/frontier-foul-play
ENV FOUL_PLAY_PYTHON=/opt/frontier-foul-play/bin/python
