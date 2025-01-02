#include <stdio.h>
#include <limits.h>
#include <stdlib.h>

void checkPointerOverflow(int *covmat) {
    int a = INT_MAX;
    printf("Value of a: %d\n", a); 

    // Check for overflow (e.g., assigning a value beyond UINT_MAX)
    int b = a + 100; // overflow but no core dumped, only warning
    printf("Value of b (may overflow): %d\n", b); 
    *(covmat + 100) = 3; // This may cause memory issues due to overflow
    // free(p);
}

int main() {
    int rows = 47000;
    int cols = 47000;

    int *covmat = (int *)malloc(rows * cols * sizeof(int));

    if (covmat == NULL) {
        printf("Memory allocation failed\n");
        return 1; // Exit if memory allocation fails
    }

    // Initialize the matrix (optional)
    for (int i = 0; i < rows * cols; i++) {
        covmat[i] = 0.0;
    }

    // Call the function to check pointer overflow
    checkPointerOverflow(covmat);

    // Free the allocated memory
    free(covmat);

    return 0;
}

