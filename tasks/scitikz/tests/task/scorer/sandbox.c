/* Root-only launcher. Candidate TeX sees only a credential-free toolchain and its own job. */
#define _GNU_SOURCE
#include <errno.h>
#include <grp.h>
#include <sched.h>
#include <seccomp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <unistd.h>

static void fail(const char *what) { perror(what); exit(125); }
static void limit(int kind, rlim_t value) {
    struct rlimit r = {value, value};
    if (setrlimit(kind, &r)) fail("setrlimit");
}
int main(int argc, char **argv) {
    if (argc < 5 || geteuid() != 0) return 125;
    char *end;
    unsigned long uid = strtoul(argv[2], &end, 10);
    if (*end || uid < 20000 || uid > 60000) return 125;
    limit(RLIMIT_AS, (rlim_t)2 * 1024 * 1024 * 1024);
    limit(RLIMIT_CPU, 30);
    limit(RLIMIT_NPROC, 32);
    limit(RLIMIT_NOFILE, 128);
    /* Ghostscript uses uncompressed temporary rasters: 32 megapixels need >128 MiB. */
    limit(RLIMIT_FSIZE, 256 * 1024 * 1024);
    limit(RLIMIT_CORE, 0);
    if (chroot(argv[1]) || chdir(argv[3])) fail("chroot/chdir");
    if (setgroups(0, NULL) || setresgid(uid, uid, uid) || setresuid(uid, uid, uid)) fail("drop identity");
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) fail("no_new_privs");
    if (clearenv() || setenv("PATH", "/usr/bin:/bin", 1) || setenv("HOME", argv[3], 1) ||
        setenv("TMPDIR", argv[3], 1) || setenv("LANG", "C.UTF-8", 1) ||
        setenv("openin_any", "p", 1) || setenv("openout_any", "p", 1) ||
        setenv("OMP_NUM_THREADS", "1", 1) || setenv("MAGICK_THREAD_LIMIT", "1", 1)) fail("environment");
    scmp_filter_ctx filter = seccomp_init(SCMP_ACT_ERRNO(EPERM));
    if (!filter) fail("seccomp_init");
    const char *allowed[] = {
        "read", "write", "readv", "writev", "pread64", "pwrite64", "open", "openat", "close", "close_range",
        "stat", "lstat", "fstat", "newfstatat", "statx", "statfs", "fstatfs", "access", "faccessat", "faccessat2",
        "readlink", "readlinkat", "lseek", "getdents", "getdents64", "getcwd", "chdir", "fchdir",
        "mmap", "mprotect", "munmap", "mremap", "madvise", "brk", "arch_prctl",
        "set_tid_address", "set_robust_list", "rseq", "futex", "futex_waitv",
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "sigaltstack", "tgkill", "kill",
        "getpid", "gettid", "getppid", "getuid", "geteuid", "getgid", "getegid", "getgroups",
        "uname", "prlimit64", "getrlimit", "getrusage", "clock_gettime", "clock_getres",
        "clock_nanosleep", "nanosleep", "gettimeofday", "time", "sysinfo", "getrandom",
        "sched_yield", "sched_getaffinity", "fcntl", "dup", "dup2", "dup3", "ioctl",
        "pipe", "pipe2", "poll", "ppoll", "select", "pselect6", "wait4", "waitid", "fork", "vfork",
        "mkdir", "mkdirat", "rmdir", "unlink", "unlinkat", "rename", "renameat", "renameat2",
        "chmod", "fchmod", "fchmodat", "umask", "utime", "utimes", "utimensat", "ftruncate", "truncate", "fsync",
        "execve", "exit", "exit_group", "restart_syscall", NULL
    };
    for (int i = 0; allowed[i]; ++i) {
        int nr = seccomp_syscall_resolve_name(allowed[i]);
        if (nr != __NR_SCMP_ERROR && seccomp_rule_add(filter, SCMP_ACT_ALLOW, nr, 0)) fail("allow syscall");
    }
    if (seccomp_rule_add(filter, SCMP_ACT_ERRNO(ENOSYS), SCMP_SYS(clone3), 0)) fail("clone3");
    unsigned long namespaces = CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWIPC | CLONE_NEWUTS | CLONE_NEWCGROUP;
    if (seccomp_rule_add(filter, SCMP_ACT_ALLOW, SCMP_SYS(clone), 1,
                        SCMP_A0(SCMP_CMP_MASKED_EQ, namespaces, 0))) fail("clone");
    if (seccomp_load(filter)) fail("load seccomp");
    seccomp_release(filter);
    execv(argv[4], &argv[4]);
    fail("execv");
}
