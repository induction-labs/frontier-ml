#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "isolation check failed: %s\n", #x); return 1; } } while (0)
static void *thread(void *arg) { return arg; }
int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--memory") == 0) {
        /* Reserve, but never touch, excess private memory. Linux rejects this
           through RLIMIT_DATA. Modal currently needs the parent watchdog. */
        void *memory = mmap(NULL, (size_t)4 * 1024 * 1024 * 1024 + 4096, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory == MAP_FAILED && errno == ENOMEM) {
            puts("FRONTIER_MEMORY_OK");
            return 0;
        }
        REQUIRE(memory != MAP_FAILED);
        sleep(3);
        fprintf(stderr, "memory watchdog did not terminate excess allocation\n");
        return 1;
    }
    struct rlimit r;
    cpu_set_t cpus;
    REQUIRE(getuid() >= 10000 && geteuid() == getuid());
    REQUIRE(prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) == 1);
    REQUIRE(prctl(PR_GET_SECCOMP, 0, 0, 0, 0) == 2);
    REQUIRE(getenv("FRONTIER_CHECKER_SECRET_CANARY") == NULL);
    REQUIRE(getrlimit(RLIMIT_DATA, &r) == 0 && r.rlim_max == (rlim_t)4 * 1024 * 1024 * 1024);
    REQUIRE(getrlimit(RLIMIT_AS, &r) == 0 && r.rlim_max == (rlim_t)32 * 1024 * 1024 * 1024);
    REQUIRE(sched_getaffinity(0, sizeof(cpus), &cpus) == 0 && CPU_COUNT(&cpus) == 1);
    REQUIRE(socket(AF_INET, SOCK_STREAM, 0) == -1 && errno == EPERM);
    REQUIRE(fork() == -1 && errno == EPERM);
    REQUIRE(open("/tests/config.json", O_RDONLY) == -1 && errno == ENOENT);
    REQUIRE(open("/proc/self/environ", O_RDONLY) == -1 && errno == ENOENT);
    REQUIRE(open("/escape", O_WRONLY | O_CREAT, 0600) == -1 && errno == EACCES);
    REQUIRE(open("/opt/mathlib/lean-toolchain", O_WRONLY) == -1 && errno == EACCES);
    REQUIRE(setuid(0) == -1);
    pthread_t t;
    REQUIRE(pthread_create(&t, NULL, thread, NULL) == 0);
    REQUIRE(pthread_join(t, NULL) == 0);
    puts("FRONTIER_SANDBOX_OK");
    return 0;
}
