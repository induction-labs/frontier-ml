"""Privileged namespace setup; no candidate code executes before privileges drop."""

import ctypes
import fcntl
import os
import socket
import struct
import sys


def main():
    submission, work = sys.argv[1:]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        flags = fcntl.ioctl(sock, 0x8913, struct.pack("16sH14s", b"lo", 0, b""))
        value = struct.unpack("16sH14s", flags)[1]
        fcntl.ioctl(sock, 0x8914, struct.pack("16sH14s", b"lo", value | 1, b""))
    os.chdir(submission)
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    # PR_SET_NO_NEW_PRIVS: setuid binaries/file capabilities cannot regain privileges.
    if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "Unable to set no_new_privs")
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOST": "0.0.0.0", "PORT": "8000",
           "MODEL_NAME": "submission", "HOME": work, "TMPDIR": work,
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HOME": work + "/huggingface", "PYTHONDONTWRITEBYTECODE": "1"}
    # CUDA runtime paths are administrator configuration, never credentials.
    for key in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if key in os.environ:
            env[key] = os.environ[key]
    os.execve("/bin/bash", ["bash", os.path.join(submission, "serve.sh")], env)


if __name__ == "__main__":
    main()
