#include <iostream>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
using std::vector;
using std::srand;

namespace ICA {

    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    Matrix MixMatrix(int size){
      return Matrix::Random(size, size);
    };

    struct FastICAResult {
        Matrix Y;
    };

    FastICAResult FastICA(const Matrix X) {
        const auto sample = X.rows();
        const auto series = X.cols();
        
        const Matrix X_center = X.colwise() - X.rowwise().mean();
        const Matrix X_cov = (X_center * X_center.transpose()) / double(X_center.cols() - 1);

        Eigen::SelfAdjointEigenSolver<Matrix> es(X_cov);
        if (es.info() != Eigen::Success) abort();

        Vector lambdas = es.eigenvalues().real();
        Matrix P = es.eigenvectors().real();
        Matrix Atilda = lambdas.cwiseSqrt().asDiagonal().inverse() * P.transpose();
        Matrix X_whiten = Atilda * X_center;

        std::cout << X_whiten << std::endl;
        
        // const auto eval = X_whiten.data();
        return FastICAResult{X};
    };
}

int main(){
  srand(0);
  Eigen::MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;

  const ICA::Matrix mmm = ICA::Matrix::Random(2, 3);
  ICA::FastICA(mmm);
  return 0;
}