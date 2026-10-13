FROM python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev libseccomp-dev texlive-latex-extra=2024.20250309-2 texlive-pictures=2024.20250309-1 texlive-science=2024.20250309-2 \
    texlive-fonts-recommended=2024.20250309-1 texlive-extra-utils=2024.20250309-2 imagemagick=8:7.1.1.43+dfsg1-1+deb13u12 ghostscript=10.05.1~dfsg-1+deb13u1 texlive-fonts-extra=2024.20250309-2 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN sed -i 's/rights="none" pattern="PDF"/rights="read" pattern="PDF"/' /etc/ImageMagick*/policy.xml
COPY scorer/sandbox.c scorer/build_texroot.py /opt/scitikz/
RUN gcc -O2 -Wall -Wextra -Werror /opt/scitikz/sandbox.c -lseccomp -o /opt/scitikz/sandbox \
    && python /opt/scitikz/build_texroot.py
RUN pip install --no-cache-dir pillow==11.3.0
