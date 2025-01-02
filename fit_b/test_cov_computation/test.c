#include <stdio.h>
#include <limits.h>
#include <stdlib.h>

void checkPointerOverflow(double *covmat) {
    int a = INT_MAX;
    printf("Value of a: %d\n", a);

    // Check for overflow (e.g., assigning a value beyond UINT_MAX)
    int b = a + 100; // overflow but no core dumped
    printf("Value of b (may overflow): %d\n", b);

    *(covmat+b)=3;
    /* free(p); */
}


