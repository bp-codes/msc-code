

Matrix<double> dgemm_serial(double alpha,
                            const Matrix<double>& A,
                            const Matrix<double>& B,
                            double beta,
                            const Matrix<double>& C)
{
    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    if (B.rows() != K)
        throw std::invalid_argument("B.rows() must equal A.cols()");
    if (C.rows() != M || C.cols() != N)
        throw std::invalid_argument("C must be M x N");

    Matrix<double> X(M, N);

    // X = beta * C
    for (std::size_t i = 0; i < M; ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            X(i, j) = beta * C(i, j);
        }
    }

    constexpr std::size_t BLOCK = 64;

    const std::size_t lda = A.stride();
    const std::size_t ldb = B.stride();
    const std::size_t ldx = X.stride();

    const double* a_data = A.data();
    const double* b_data = B.data();
    double* x_data = X.data();

    for (std::size_t ib = 0; ib < M; ib += BLOCK)
    {
        for (std::size_t kb = 0; kb < K; kb += BLOCK)
        {
            for (std::size_t jb = 0; jb < N; jb += BLOCK)
            {
                const std::size_t i_max = std::min(ib + BLOCK, M);
                const std::size_t k_max = std::min(kb + BLOCK, K);
                const std::size_t j_max = std::min(jb + BLOCK, N);

                for (std::size_t i = ib; i < i_max; ++i)
                {
                    for (std::size_t k = kb; k < k_max; ++k)
                    {
                        const double a_ik = alpha * a_data[i * lda + k];
                        const __m256d a_vec = _mm256_set1_pd(a_ik);

                        std::size_t j = jb;

                        // AVX: 4 doubles at a time
                        for (; j + 4 <= j_max; j += 4)
                        {
                            const __m256d b_vec =
                                _mm256_loadu_pd(&b_data[k * ldb + j]);

                            __m256d x_vec =
                                _mm256_loadu_pd(&x_data[i * ldx + j]);

                            x_vec = _mm256_fmadd_pd(a_vec, b_vec, x_vec);

                            _mm256_storeu_pd(&x_data[i * ldx + j], x_vec);
                        }

                        // Scalar tail
                        for (; j < j_max; ++j)
                        {
                            x_data[i * ldx + j] += a_ik * b_data[k * ldb + j];
                        }
                    }
                }
            }
        }
    }

    return X;
}




























