#include <stdio.h>
#include <limits.h>

int main() {
    printf("Size of long: %zu bytes\n", sizeof(long));
    printf("Size of long long: %zu bytes\n", sizeof(long long));
    printf("Range of signed long: %ld to %ld\n", LONG_MIN, LONG_MAX);
    printf("Range of signed long long: %lld to %lld\n", LLONG_MIN, LLONG_MAX);
    return 0;
}

