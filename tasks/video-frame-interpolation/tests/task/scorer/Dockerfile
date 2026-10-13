FROM python:3.12-slim
# Weight acquisition happens at image build, never during grading. All downloads
# must match the organizer's committed lock; the draft lock deliberately fails.
COPY pipeline/metric_bundle.py /opt/vfi-build/metric_bundle.py
COPY scorer/metric-lock.json /opt/vfi-build/metric-lock.json
RUN python /opt/vfi-build/metric_bundle.py restore --lock /opt/vfi-build/metric-lock.json --output /opt/vfi-metrics
# The common verifier layers install torch==2.8.0 (CPU), numpy==2.2.6, Pillow==11.3.0.
# This parent validates hashes and orchestrates the run. Actual quality metrics
# run in Dockerfile.gpu after the candidate H100 has been destroyed.
