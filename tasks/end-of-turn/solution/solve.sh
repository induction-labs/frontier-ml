#!/bin/bash
set -euo pipefail
mkdir -p /workspace/output
cp /solution/server.py /workspace/output/server.py
cat > /workspace/output/serve.sh <<'SH'
#!/bin/bash
set -euo pipefail
exec python "$(dirname "$0")/server.py"
SH
chmod +x /workspace/output/serve.sh
