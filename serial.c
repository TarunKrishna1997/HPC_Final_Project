#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to multiply matrices
void matrix_multiply_serial(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

// Main function to demonstrate matrix multiplication
int main() {
    int n = 1000; // Large matrix size
    double *A, *B, *C;

    // Allocate memory
    A = (double *)malloc(n * n * sizeof(double));
    B = (double *)malloc(n * n * sizeof(double));
    C = (double *)malloc(n * n * sizeof(double));

    // Initialize matrices with random values
    for (int i = 0; i < n*n; i++) {
        A[i] = drand48();
        B[i] = drand48();
    }

    // Measure time
    clock_t start = clock();
    matrix_multiply_serial(A, B, C, n);
    clock_t end = clock();

    printf("Time taken: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Cleanup
    free(A);
    free(B);
    free(C);

    return 0;
}
