#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void matrix_multiply_parallel(double *A, double *B, double *C, int n, int local_rows) {
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    int n = 1000; 
    int rank, size, local_rows;
    double *A, *B, *C, *local_A, *local_C;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_rows = n / size; 

    
    B = (double *)malloc(n * n * sizeof(double));
    local_A = (double *)malloc(local_rows * n * sizeof(double));
    local_C = (double *)calloc(local_rows * n, sizeof(double)); 

    if (rank == 0) {
        A = (double *)malloc(n * n * sizeof(double));
        C = (double *)calloc(n * n, sizeof(double)); 

        
        srand48(time(NULL)); 
        for (int i = 0; i < n * n; i++) {
            A[i] = drand48();
            B[i] = drand48();
        }
    }

    // Timing start
    double start_time = MPI_Wtime();

    // Distribute A and broadcast B
    MPI_Scatter(A, local_rows * n, MPI_DOUBLE, local_A, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix_multiply_parallel(local_A, B, local_C, n, local_rows);

    MPI_Gather(local_C, local_rows * n, MPI_DOUBLE, C, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Timing end
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Matrix multiplication completed in %f seconds\n", end_time - start_time);
    }

    // Cleanup
    free(B);
    free(local_A);
    free(local_C);
    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
}
