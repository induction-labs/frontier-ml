#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
printf '0\n' > /logs/verifier/reward.txt
chmod 700 /tests /logs/verifier
export PYTHONPATH=/tests/lib
# Prepare the task's private evaluation data before any inference machine starts.
python -m spin.transfer fetch /tests/data.json /tests/private
# Harbor transfers /workspace/output to the same path in the separate verifier.
# Trusted CPU verifier provisions offline inference through Harbor environments.
# Only the submission goes to that GPU; labels and judge credentials stay here.
python /tests/grade.py
