FROM python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c
# Debian removes superseded security builds, so the pins below resolve against a fixed snapshot.
ARG DEBIAN_SNAPSHOT=20261009T000000Z
RUN printf 'Types: deb\nURIs: http://snapshot.debian.org/archive/debian/%s\nSuites: trixie trixie-updates\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n\nTypes: deb\nURIs: http://snapshot.debian.org/archive/debian-security/%s\nSuites: trixie-security\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n' "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\nAcquire::Retries "5";\n' > /etc/apt/apt.conf.d/99snapshot
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git gcc libc6-dev libseccomp-dev texlive-latex-extra=2024.20250309-2 texlive-pictures=2024.20250309-1 texlive-science=2024.20250309-2 \
    texlive-fonts-recommended=2024.20250309-1 texlive-extra-utils=2024.20250309-2 imagemagick=8:7.1.1.43+dfsg1-1+deb13u12 ghostscript=10.05.1~dfsg-1+deb13u2 texlive-fonts-extra=2024.20250309-2 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN sed -i 's/rights="none" pattern="PDF"/rights="read" pattern="PDF"/' /etc/ImageMagick*/policy.xml
COPY scorer/sandbox.c scorer/build_texroot.py /opt/scitikz/
RUN gcc -O2 -Wall -Wextra -Werror /opt/scitikz/sandbox.c -lseccomp -o /opt/scitikz/sandbox \
    && python /opt/scitikz/build_texroot.py
ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
COPY scorer/requirements.txt /opt/scitikz/requirements.txt
RUN python -m venv /opt/scitikz/venv && /opt/scitikz/venv/bin/pip install --no-cache-dir -r /opt/scitikz/requirements.txt
COPY scitikz_models.py scorer/weights.json /opt/scitikz/
RUN /opt/scitikz/venv/bin/python /opt/scitikz/scitikz_models.py /opt/scitikz/models --expected /opt/scitikz/weights.json
WORKDIR /tmp
