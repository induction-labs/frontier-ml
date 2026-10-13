/* Narrow, non-setuid launcher for the trusted verifier's Lean subprocesses. */
#define _GNU_SOURCE
#include <errno.h>
#include <grp.h>
#include <sched.h>
#include <seccomp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

static void fail(const char *what) { perror(what); exit(125); }
static void limit(int kind, rlim_t value) {
    struct rlimit r = {value, value};
    if (setrlimit(kind, &r)) fail("setrlimit");
}

int main(int argc, char **argv) {
    /* rootfs, uid, cpu, cwd, LEAN_PATH, executable, arguments... are operator-owned. */
    if (argc < 7 || geteuid() != 0) return 125;
    char *end;
    unsigned long uid = strtoul(argv[2], &end, 10);
    if (*end || uid < 10000 || uid > 60000) return 125;
    long cpu = strtol(argv[3], &end, 10);
    if (*end || cpu < 0 || cpu >= CPU_SETSIZE) return 125;
    cpu_set_t cpus;
    CPU_ZERO(&cpus); CPU_SET(cpu, &cpus);
    if (sched_setaffinity(0, sizeof(cpus), &cpus)) fail("sched_setaffinity");
    /* Mathlib maps more than 4 GiB of read-only .olean address space. Bound
       writable allocations separately; Lean also receives -M4096. */
    limit(RLIMIT_AS, (rlim_t)32 * 1024 * 1024 * 1024);
    limit(RLIMIT_DATA, (rlim_t)4 * 1024 * 1024 * 1024);
    limit(RLIMIT_CPU, 60);
    limit(RLIMIT_NPROC, 32);
    limit(RLIMIT_NOFILE, 128);
    limit(RLIMIT_FSIZE, 4 * 1024 * 1024);
    limit(RLIMIT_CORE, 0);
    if (chroot(argv[1]) || chdir(argv[4])) fail("chroot/chdir");
    /* Lean queries /proc/<pid>/exe to find its toolchain. Provide that one
       synthetic symlink, never mount proc or expose another process's files. */
    char procdir[80], executable[96];
    snprintf(procdir, sizeof(procdir), "/proc/%ld", (long)getpid());
    snprintf(executable, sizeof(executable), "%s/exe", procdir);
    if (mkdir(procdir, 0555) || symlink(argv[6], executable)) fail("executable location");
    if (setgroups(0, NULL) || setresgid(uid, uid, uid) || setresuid(uid, uid, uid)) fail("drop uid/gid");
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) fail("no_new_privs");
    if (clearenv() || setenv("PATH", "/opt/lean/bin", 1) || setenv("HOME", argv[4], 1)
        || setenv("TMPDIR", argv[4], 1) || setenv("LANG", "C.UTF-8", 1)
        || setenv("LEAN_PATH", argv[5], 1) || setenv("LEAN_SYSROOT", "/opt/lean", 1)
        || setenv("LD_LIBRARY_PATH", "/opt/lean/lib/lean:/opt/lean/lib", 1)) fail("environment");

    /* Default deny: no network, new processes, mounts, ptrace, or host IPC.
       Threads share the same memory accounting; fork cannot multiply the limit. */
    scmp_filter_ctx filter = seccomp_init(SCMP_ACT_ERRNO(EPERM));
    if (!filter) fail("seccomp_init");
    const char *allowed[] = {
        "read", "write", "readv", "writev", "pread64", "pwrite64", "open", "openat", "close", "close_range",
        "stat", "lstat", "fstat", "newfstatat", "statx", "statfs", "fstatfs", "access", "faccessat", "faccessat2",
        "readlink", "readlinkat", "lseek", "getdents", "getdents64", "getcwd", "chdir",
        "mmap", "mprotect", "munmap", "mremap", "madvise", "brk", "arch_prctl",
        "set_tid_address", "set_robust_list", "rseq", "futex", "futex_waitv",
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "sigaltstack", "tgkill",
        "getpid", "gettid", "getppid", "getuid", "geteuid", "getgid", "getegid",
        "uname", "prlimit64", "getrlimit", "getrusage", "clock_gettime", "clock_getres",
        "clock_nanosleep", "nanosleep", "gettimeofday", "time", "sysinfo", "getrandom",
        "sched_yield", "sched_getaffinity", "fcntl", "dup", "dup2", "dup3", "prctl",
        "execve", "exit", "exit_group", "restart_syscall",
        NULL
    };
    for (int i = 0; allowed[i]; ++i) {
        int nr = seccomp_syscall_resolve_name(allowed[i]);
        if (nr != __NR_SCMP_ERROR && seccomp_rule_add(filter, SCMP_ACT_ALLOW, nr, 0)) fail("seccomp allow");
    }
    /* glibc falls back to clone when clone3 is unavailable. Only pthread-style
       clones are allowed; shared VM alone is not sufficient. */
    if (seccomp_rule_add(filter, SCMP_ACT_ERRNO(ENOSYS), SCMP_SYS(clone3), 0)) fail("seccomp clone3");
    unsigned long flags = CLONE_THREAD | CLONE_VM | CLONE_SIGHAND;
    if (seccomp_rule_add(filter, SCMP_ACT_ALLOW, SCMP_SYS(clone), 1,
                        SCMP_A0(SCMP_CMP_MASKED_EQ, flags, flags))) fail("seccomp clone");
    if (seccomp_load(filter)) fail("seccomp_load");
    seccomp_release(filter);
    execv(argv[6], &argv[6]);
    fail("execv");
}
