"""Userspace network guard for grading sandboxes whose provider does not cut the network device off.

Installed as root before the candidate's server starts:
- /etc/nsswitch.conf resolves host names from /etc/hosts only, so lookups of outside names fail at once while localhost
  still resolves, as under block_network (where the sandbox's hostname does not resolve either). This is the layer
  that holds on Modal, where /etc/resolv.conf is a read-only mount and Modal's resolver answers even with an empty
  egress allowlist.
- /etc/resolv.conf points at 127.0.0.1, where nothing answers, wherever the provider lets it be rewritten.
- A preload library listed in /etc/ld.so.preload, which a process without root can neither edit nor escape short of
  raw syscalls or a static binary, makes connect(), sendto(), sendmsg() and sendmmsg() to any address that is not
  loopback or one of the sandbox's own interface addresses fail with ENETUNREACH.

Honest code therefore sees what it saw under block_network: lookups and outbound connections fail fast, and nothing
is counted against it. A socket of the candidate's UID that still reaches for a non-local peer got past the guard
(raw syscalls, a static or Go binary); the relay's monitor (relay.Monitor) records it in resource.json and the
verifier logs it, without failing the grade: the provider's own egress block still drops those packets.
"""
from __future__ import annotations

import shlex
from pathlib import Path

LIBRARY = "/usr/local/lib/fml-netguard.so"
SOURCE = "/opt/frontier-grading/netguard.c"

C_SOURCE = Path(__file__).with_name("netguard.c")  # shipped with the package, compiled inside the sandbox

# glibc asks DNS only if the hosts line lists it, and assumes "dns files" when the file is missing, so write one then.
NSSWITCH = ("f=/etc/nsswitch.conf; if grep -q '^hosts:' $f 2>/dev/null; then sed -i 's/^hosts:.*/hosts: files/' $f; "
            "else echo 'hosts: files' >> $f; fi && chown 0:0 $f && chmod 0644 $f")

# Run as the candidate's UID after installing: an outbound connect must fail fast, a loopback one must work.
VERIFY = r'''
setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs env -i PATH=/usr/bin:/bin bash -c '
if timeout 5 bash -c "exec 3<>/dev/tcp/1.1.1.1/80" 2>/dev/null; then echo OUTBOUND_OPEN; else echo OUTBOUND_BLOCKED; fi
(exec 3<>/dev/tcp/127.0.0.1/1) 2>&1 | grep -q "refused" && echo LOOPBACK_OK || echo LOOPBACK_ODD
getent hosts example.com >/dev/null 2>&1 && echo DNS_OPEN || echo DNS_BLOCKED'
'''


def install(resource):
    """Install the guard in the grading sandbox (as root); return what was put in place, for the resource record."""
    record = {"nsswitch": False, "resolv_conf": False, "preload": False, "verified": None}
    record["nsswitch"] = resource.exec(NSSWITCH, timeout=30, check=False).return_code == 0
    result = resource.exec("cp -a /etc/resolv.conf /etc/resolv.conf.fml-orig 2>/dev/null; "
                           "printf 'nameserver 127.0.0.1\\noptions timeout:1 attempts:1\\n' > /etc/resolv.conf && "
                           "chmod 0644 /etc/resolv.conf && chown 0:0 /etc/resolv.conf", timeout=30, check=False)
    record["resolv_conf"] = result.return_code == 0
    compiler = (resource.exec("command -v gcc || command -v cc || true", timeout=30, check=False).stdout or "").strip()
    if compiler:
        resource.upload(C_SOURCE, SOURCE)
        built = resource.exec(f"{shlex.quote(compiler.splitlines()[-1])} -O2 -shared -fPIC -o {LIBRARY} {SOURCE} -ldl -lpthread && "
                              f"chown 0:0 {LIBRARY} && chmod 0644 {LIBRARY} && echo {LIBRARY} > /etc/ld.so.preload && "
                              "chown 0:0 /etc/ld.so.preload && chmod 0644 /etc/ld.so.preload", timeout=180, check=False)
        record["preload"] = built.return_code == 0
        if not record["preload"]:
            record["preload_error"] = (built.stderr or built.stdout or "")[-500:]
    else:
        record["preload_error"] = "no C compiler in the image; only name lookups are disabled"
    check = resource.exec(VERIFY, timeout=60, check=False)
    lines = (check.stdout or "").split()
    record["verified"] = {"outbound_blocked": "OUTBOUND_BLOCKED" in lines, "loopback_ok": "LOOPBACK_OK" in lines,
                          "dns_blocked": "DNS_BLOCKED" in lines}
    return record
