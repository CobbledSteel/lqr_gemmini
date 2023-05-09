#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <Eigen/Dense>
#include "gemmini.h"

using namespace Eigen;

static uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;

    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbff8);
    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbffc);
    // return *mtime;
}

// Function to solve the finite-horizon discrete time LQR problem using Riccati recursion
Matrix<float, Dynamic, Dynamic, RowMajor> lqrSolveFiniteHorizon(const Matrix<float, Dynamic, Dynamic, RowMajor>& A, const Matrix<float, Dynamic, Dynamic, RowMajor>& B, const Matrix<float, Dynamic, Dynamic, RowMajor>& Q, const Matrix<float, Dynamic, Dynamic, RowMajor>& R, int horizon) {
    int state_dim = A.rows();
    int input_dim = B.cols();

    // Initialize the gain matrices
    std::vector<Matrix<float, Dynamic, Dynamic, RowMajor>> K(horizon, Matrix<float, Dynamic, Dynamic, RowMajor>::Zero(input_dim, state_dim));

    // Initialize the value function matrix
    Matrix<float, Dynamic, Dynamic, RowMajor> P = Q;

    // Perform the Riccati recursion
    for (int t = horizon - 1; t >= 0; --t) {
        Matrix<float, Dynamic, Dynamic, RowMajor> BRB = B.transpose() * P * B + R;
        Matrix<float, Dynamic, Dynamic, RowMajor> BRB_inv = BRB.inverse();
        K[t] = BRB_inv * B.transpose() * P * A;
        P = A.transpose() * P * A - A.transpose() * P * B * K[t] + Q;
    }

    return K[0];
}

Matrix<float, Dynamic, Dynamic, RowMajor> lqrSolveFiniteHorizonSingleOp(const Matrix<float, Dynamic, Dynamic, RowMajor>& A, const Matrix<float, Dynamic, Dynamic, RowMajor>& B, const Matrix<float, Dynamic, Dynamic, RowMajor>& Q, const Matrix<float, Dynamic, Dynamic, RowMajor>& R, int horizon) {
    const int state_dim = A.rows();
    const int input_dim = B.cols();

    // Initialize the gain matrices
    std::vector<Matrix<float, Dynamic, Dynamic, RowMajor>> K(horizon, Matrix<float, Dynamic, Dynamic, RowMajor>::Zero(input_dim, state_dim));

    // Initialize the value function matrix
    Matrix<float, Dynamic, Dynamic, RowMajor> P(state_dim, state_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> PA(state_dim, state_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> PB(state_dim, input_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> BPB_R(input_dim, input_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> BPA(input_dim, state_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> APA(state_dim, state_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> BPB_R_inv(input_dim, input_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> BPAK_Q(state_dim, state_dim);
    // std::cout << "P rows: " << P.rows() << ", cols: " << P.cols() << std::endl;
    // std::cout << "A rows: " << A.rows() << ", cols: " << A.cols() << std::endl;
    // std::cout << "B rows: " << B.rows() << ", cols: " << B.cols() << std::endl;

    P = Q;
    // Perform the Riccati recursion
    for (int t = horizon - 1; t >= 0; --t) {
        PA = P * A;
        PB = P * B;
        BPB_R = B.transpose() * PB + R;
        BPA = B.transpose() * PA;
        APA = A.transpose() * PA;
        BPB_R_inv = BPB_R.inverse(); // CPU
        K[t] = BPB_R_inv * BPA;
        BPAK_Q = BPA.transpose() * K[t] + Q;
        P = APA - BPAK_Q;
    }
    return K[0];
}

void tiled_matmul_auto_eigen (
    const Matrix<float, Dynamic, Dynamic, RowMajor>&A,
    const Matrix<float, Dynamic, Dynamic, RowMajor>&B,
    Matrix<float, Dynamic, Dynamic, RowMajor>&C,
    bool transpose_A, bool transpose_B) 
{
        int i = transpose_A ? A.cols() : A.rows();
        int j = transpose_B ? B.rows() : B.cols();
        int k = transpose_B ? B.cols() : B.rows();
        tiled_matmul_auto(i, j, k,
                A.data(), B.data(), NULL, C.data(),
                k, j, j, j,
                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                transpose_A, transpose_B,
                false, false,
                0,
                WS
                );
}

void tiled_matmul_auto_eigen_bias (
    const Matrix<float, Dynamic, Dynamic, RowMajor>&A,
    const Matrix<float, Dynamic, Dynamic, RowMajor>&B,
    const Matrix<float, Dynamic, Dynamic, RowMajor>&D,
    Matrix<float, Dynamic, Dynamic, RowMajor>&C,
    bool transpose_A,
    bool transpose_B,
    bool sub ) 
{
        int i = transpose_A ? A.cols() : A.rows();
        int j = transpose_B ? B.rows() : B.cols();
        int k = transpose_B ? B.cols() : B.rows();
        tiled_matmul_auto(i, j, k,
                A.data(), B.data(), D.data() , C.data(),
                k, j, j, j,
                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, sub ? -MVIN_SCALE_IDENTITY : MVIN_SCALE_IDENTITY,
                NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                transpose_A, transpose_B,
                false, false,
                0,
                WS
                );
}

Matrix<float, Dynamic, Dynamic, RowMajor> lqrSolveFiniteHorizonGemminiC(const Matrix<float, Dynamic, Dynamic, RowMajor>& A, const Matrix<float, Dynamic, Dynamic, RowMajor>& B, const Matrix<float, Dynamic, Dynamic, RowMajor>& Q, const Matrix<float, Dynamic, Dynamic, RowMajor>& R, int horizon) {
    const int state_dim = A.rows();
    const int input_dim = B.cols();

    // Initialize the gain matrices
    std::vector<Matrix<float, Dynamic, Dynamic, RowMajor>> K(horizon, Matrix<float, Dynamic, Dynamic, RowMajor>::Zero(input_dim, state_dim));

    Matrix<float, Dynamic, Dynamic, RowMajor> BT(input_dim, state_dim);
    BT = B.transpose().eval();
    Matrix<float, Dynamic, Dynamic, RowMajor> AT(input_dim, state_dim);
    AT = A.transpose().eval();
    Matrix<float, Dynamic, Dynamic, RowMajor> BPAT(state_dim, input_dim);

    // Initialize the value function matrix
    Matrix<float, Dynamic, Dynamic, RowMajor> P(state_dim, state_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> PA(state_dim, state_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> PB(state_dim, input_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> BPB_R(input_dim, input_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> BPA(input_dim, state_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> APA(state_dim, state_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> BPB_R_inv(input_dim, input_dim);
    Matrix<float, Dynamic, Dynamic, RowMajor> BPAK_Q(state_dim, state_dim);
    // std::cout << "P rows: " << P.rows() << ", cols: " << P.cols() << std::endl;
    // std::cout << "A rows: " << A.rows() << ", cols: " << A.cols() << std::endl;
    // std::cout << "B rows: " << B.rows() << ", cols: " << B.cols() << std::endl;

    P = Q;
    // Perform the Riccati recursion
    for (int t = horizon - 1; t >= 0; --t) {
        // PA = P * A;
        tiled_matmul_auto_eigen(P, A, PA, false, false);
        tiled_matmul_auto_eigen(P, B, PB, false, false);
        tiled_matmul_auto_eigen_bias(BT, PB, R, BPB_R, false, false, false);
        tiled_matmul_auto_eigen(BT, PA, BPA, false, false);
        tiled_matmul_auto_eigen(AT, PA, APA, false, false);
        BPB_R_inv = BPB_R.inverse(); // CPU
        tiled_matmul_auto_eigen(BPB_R_inv, BPA, K[t], false, false);
        BPAT = BPA.transpose().eval();
        tiled_matmul_auto_eigen_bias(BPAT, K[t], Q, BPAK_Q, false, false, false);
        tiled_matmul_auto_eigen_bias(AT, PA, BPAK_Q, P, false, false, true);
        // P = APA - BPAK_Q;
    }
    return K[0];
}
// // This function runs a tiled matrix multiplication, with automatically
// // calculated tiling factors
// static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
//         const elem_t* A, const elem_t* B,
//         const void * D, void * C,
//         size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
//         scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
//         int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
//         bool transpose_A, bool transpose_B,
//         bool full_C, bool low_D,
//         uint8_t weightA,
//         enum tiled_matmul_type_t tiled_matmul_type) {

// static void tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
//         const elem_t* A, const elem_t* B,
//         const void * D, void* C,
//         size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
//         scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
//         int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
//         size_t tile_I, size_t tile_J, size_t tile_K,
//         bool transpose_A, bool transpose_B,
//         bool full_C, bool low_D,
//         uint8_t weightA,
//         enum tiled_matmul_type_t tiled_matmul_type) {

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <state_space_size> <action_space_size> <horizon_length>" << std::endl;
        return 1;
    }

    int state_dim = std::stoi(argv[1]);
    int input_dim = std::stoi(argv[2]);
    int horizon   = std::stoi(argv[3]);

    uint64_t t0;
    uint64_t t1;
    uint64_t time1;
    uint64_t time2;
    uint64_t time3;

    // Initialize the random number generator
    std::srand(static_cast<unsigned>(std::time(0)));

    // Define the system matrices A and B with random values
    Matrix<float, Dynamic, Dynamic, RowMajor> A(state_dim, state_dim);
    A.setRandom();
    Matrix<float, Dynamic, Dynamic, RowMajor> B(state_dim, input_dim);
    B.setRandom();

    // Define the cost matrices Q and R with random values
    Matrix<float, Dynamic, Dynamic, RowMajor> Q(state_dim, state_dim);
    Q.setRandom();
    Q = Q.transpose() * Q; // Make Q symmetric and positive semi-definite
    Matrix<float, Dynamic, Dynamic, RowMajor> R(input_dim, input_dim);
    R.setRandom();
    R = R.transpose() * R; // Make R symmetric and positive definite

    // Solve the finite-horizon LQR problem
    t0 = read_cycles();
    Matrix<float, Dynamic, Dynamic, RowMajor> K = lqrSolveFiniteHorizon(A, B, Q, R, horizon);
    t1 = read_cycles();
    time1 = t1 - t0;

    t0 = read_cycles();
    Matrix<float, Dynamic, Dynamic, RowMajor> K2 = lqrSolveFiniteHorizonSingleOp(A, B, Q, R, horizon);
    t1 = read_cycles();
    time2 = t1 - t0;
    t0 = read_cycles();

    t0 = read_cycles();
    Matrix<float, Dynamic, Dynamic, RowMajor> K3 = lqrSolveFiniteHorizonGemminiC(A, B, Q, R, horizon);
    t1 = read_cycles();
    time3 = t1 - t0;

    std::cout << "Optimal gain matrix K:  (" << time1 << " )" << std::endl;
    std::cout << K << std::endl;

    std::cout << "Optimal gain matrix K2: (" << time2 << " )" << std::endl;
    // std::cout << K2 << std::endl;

    std::cout << "Optimal gain matrix K3: (" << time3 << " )" << std::endl;
    std::cout << K3 << std::endl;

    return 0;
}
