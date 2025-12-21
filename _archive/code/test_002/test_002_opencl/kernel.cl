__kernel void matmul(
    __global const double* A,
    __global const double* B,
    __global double* C,
    int N) {

    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}