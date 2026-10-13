#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>

static int (*real_connect)(int, const struct sockaddr *, socklen_t);
static ssize_t (*real_sendto)(int, const void *, size_t, int, const struct sockaddr *, socklen_t);
static ssize_t (*real_sendmsg)(int, const struct msghdr *, int);
static int (*real_sendmmsg)(int, struct mmsghdr *, unsigned int, int);
static struct in_addr local4[64];
static struct in6_addr local6[64];
static int n4, n6;
static pthread_once_t once = PTHREAD_ONCE_INIT;

static void init(void) {
    real_connect = dlsym(RTLD_NEXT, "connect");
    real_sendto = dlsym(RTLD_NEXT, "sendto");
    real_sendmsg = dlsym(RTLD_NEXT, "sendmsg");
    real_sendmmsg = dlsym(RTLD_NEXT, "sendmmsg");
    struct ifaddrs *list, *it;
    if (getifaddrs(&list) == 0) {
        for (it = list; it; it = it->ifa_next) {
            if (!it->ifa_addr) continue;
            if (it->ifa_addr->sa_family == AF_INET && n4 < 64)
                local4[n4++] = ((struct sockaddr_in *)it->ifa_addr)->sin_addr;
            else if (it->ifa_addr->sa_family == AF_INET6 && n6 < 64)
                local6[n6++] = ((struct sockaddr_in6 *)it->ifa_addr)->sin6_addr;
        }
        freeifaddrs(list);
    }
}

static int local_v4(struct in_addr address) {
    uint32_t host = ntohl(address.s_addr);
    if ((host >> 24) == 127 || host == 0) return 1;
    for (int i = 0; i < n4; i++) if (local4[i].s_addr == address.s_addr) return 1;
    return 0;
}

static int allowed(const struct sockaddr *address) {
    pthread_once(&once, init);
    if (!address) return 1;
    if (address->sa_family == AF_INET) return local_v4(((const struct sockaddr_in *)address)->sin_addr);
    if (address->sa_family == AF_INET6) {
        const struct in6_addr *a = &((const struct sockaddr_in6 *)address)->sin6_addr;
        if (IN6_IS_ADDR_LOOPBACK(a) || IN6_IS_ADDR_UNSPECIFIED(a)) return 1;
        if (IN6_IS_ADDR_V4MAPPED(a)) { struct in_addr v4; memcpy(&v4, &a->s6_addr[12], 4); return local_v4(v4); }
        for (int i = 0; i < n6; i++) if (!memcmp(&local6[i], a, sizeof *a)) return 1;
        return 0;
    }
    return 1;  /* AF_UNIX, AF_NETLINK, AF_UNSPEC (disconnect) */
}

int connect(int fd, const struct sockaddr *address, socklen_t length) {
    if (!allowed(address)) { errno = ENETUNREACH; return -1; }
    return real_connect(fd, address, length);
}

ssize_t sendto(int fd, const void *buffer, size_t size, int flags, const struct sockaddr *address, socklen_t length) {
    if (address && !allowed(address)) { errno = ENETUNREACH; return -1; }
    return real_sendto(fd, buffer, size, flags, address, length);
}

ssize_t sendmsg(int fd, const struct msghdr *message, int flags) {
    if (message && message->msg_name && !allowed((const struct sockaddr *)message->msg_name)) { errno = ENETUNREACH; return -1; }
    return real_sendmsg(fd, message, flags);
}

int sendmmsg(int fd, struct mmsghdr *messages, unsigned int count, int flags) {
    for (unsigned int i = 0; i < count; i++)
        if (messages[i].msg_hdr.msg_name && !allowed((const struct sockaddr *)messages[i].msg_hdr.msg_name)) {
            errno = ENETUNREACH; return -1;
        }
    return real_sendmmsg(fd, messages, count, flags);
}
