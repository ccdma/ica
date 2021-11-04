#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <eigen3/Eigen/Dense>
using std::vector;
using std::srand;

#define DEBUG true

namespace ICA {

    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    Matrix RandMatrix(int size){
      return Matrix::Random(size, size);
    };

    // 正方行列でなくてはいけない
    // i番目を直行空間に射影
    void Normalize(Matrix mat, int i){
      const auto size = mat.cols();
      if (i>0){
        mat.col(i) = mat.col(i) - mat.block(0, 0, size, i+1) * mat.block(0, 0, size, i+1).transpose() * mat.col(i);
      }
      mat.col(i) = mat.col(i) / mat.col(i).squaredNorm();
    }

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

        // 単位行列であることを確認
        if (DEBUG) std::cout << (X_whiten * X_whiten.transpose()) / double(X_whiten.cols() - 1) << std::endl;

        const auto g = [](double bx) { return std::pow(bx, 3); };
        const auto g2 = [](double bx) { return 3*std::pow(bx, 2); };

        const auto I = X_whiten.rows();
        const auto B = RandMatrix(I);

        for(int i=0; i<I; i++){
          std::cout << B << std::endl;
          Normalize(B, i);
        }

        if(DEBUG) std::cout << B * B.transpose() << std::endl;
        
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