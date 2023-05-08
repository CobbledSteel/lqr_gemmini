#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

// Function to solve the finite-horizon discrete time LQR problem using Riccati recursion
MatrixXd lqrSolveFiniteHorizon(const MatrixXd& A, const MatrixXd& B, const MatrixXd& Q, const MatrixXd& R, int horizon) {
    int state_dim = A.rows();
    int input_dim = B.cols();

    // Initialize the gain matrices
    std::vector<MatrixXd> K(horizon, MatrixXd::Zero(input_dim, state_dim));

    // Initialize the value function matrix
    MatrixXd P = Q;

    // Perform the Riccati recursion
    for (int t = horizon - 1; t >= 0; --t) {
        MatrixXd BRB = B.transpose() * P * B + R;
        MatrixXd BRB_inv = BRB.inverse();
        K[t] = BRB_inv * B.transpose() * P * A;
        P = A.transpose() * P * A - A.transpose() * P * B * K[t] + Q;
    }

    return K[0];
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <state_space_size> <action_space_size>" << std::endl;
        return 1;
    }

    int state_dim = std::stoi(argv[1]);
    int input_dim = std::stoi(argv[2]);

    // Initialize the random number generator
    std::srand(static_cast<unsigned>(std::time(0)));

    // Define the system matrices A and B with random values
    MatrixXd A = MatrixXd::Random(state_dim, state_dim);
    MatrixXd B = MatrixXd::Random(state_dim, input_dim);

    // Define the cost matrices Q and R with random values
    MatrixXd Q = MatrixXd::Random(state_dim, state_dim);
    Q = Q.transpose() * Q; // Make Q symmetric and positive semi-definite
    MatrixXd R = MatrixXd::Random(input_dim, input_dim);
    R = R.transpose() * R; // Make R symmetric and positive definite

    // Define the control horizon
    int horizon = 5;

    // Solve the finite-horizon LQR problem
    MatrixXd K = lqrSolveFiniteHorizon(A, B, Q, R, horizon);

    std::cout << "Optimal gain matrix K:" << std::endl;
    std::cout << K << std::endl;

    return 0;
}
