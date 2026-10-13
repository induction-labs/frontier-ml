"""Build-time assembly of the verifier's small, credential-free Lean filesystem."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

root = Path('/opt/frontier-lean/rootfs')
root.mkdir(parents=True, exist_ok=True)
mathlib = Path('/opt/mathlib')
libraries = [mathlib / '.lake/build/lib', *sorted((mathlib / '.lake/packages').glob('*/.lake/build/lib'))]
subprocess.run(['/opt/lean/bin/lean', '-o', 'ProofCheck.olean', 'ProofCheck.lean'], cwd='/checker',
               env=dict(os.environ, LEAN_PATH=':'.join(map(str, libraries))), check=True)
linked = subprocess.check_output(['ldd', '/opt/lean/bin/lean'], text=True)
linked += subprocess.check_output(['ldd', '/checker/isolation-probe'], text=True)
for library in set(re.findall(r'(/\S+)', linked)):
    if library.startswith('/opt/lean/'):
        continue
    target = root / library.lstrip('/')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(library).resolve(), target)
(root / 'opt').mkdir(exist_ok=True)
for source, target in [('/opt/lean', 'opt/lean'), ('/opt/mathlib', 'opt/mathlib'), ('/checker', 'checker')]:
    shutil.move(source, root / target)
(root / 'jobs').mkdir(mode=0o711)
(root / 'proc').mkdir(mode=0o755)
(root / 'dev').mkdir(mode=0o755)
subprocess.run(['chown', '-R', '0:0', str(root)], check=True)
subprocess.run(['chmod', '-R', 'go-w', str(root)], check=True)
files = {}
for path in sorted(root.rglob('*')):
    if path.is_file():
        files[path.relative_to(root).as_posix()] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
Path('/opt/frontier-lean/build.json').write_text(json.dumps({
    'protocol': 'lean-simplification-v1', 'lean': '4.15.0',
    'mathlib': '9837ca9d65d9de6fad1ef4381750ca688774e608',
    'fingerprint': fingerprint, 'files': len(files),
    'checker_source_sha256': files['checker/ProofCheck.lean'],
    'sandbox_sha256': hashlib.sha256(Path('/opt/frontier-lean/sandbox').read_bytes()).hexdigest(),
}, indent=2) + '\n')
