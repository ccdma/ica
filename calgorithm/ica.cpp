#include <iostream>
#include <vector>
#include <random>
#include <eigen3/Eigen/Core>
using std::vector;
using std::srand;

namespace ICA {

    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    Matrix MixMatrix(int size){
      return Matrix::Random(size, size);
    };

    Matrix Cov(Matrix mat){
      Matrix centered = mat.rowwise() - mat.colwise().mean();
      Matrix cov = (centered.adjoint() * centered) / double(mat.rows() - 1);
      return cov;
    }

    struct FastICAResult {
        Matrix Y;
    };

    FastICAResult FastICA(const Matrix X) {
        const auto sample = X.rows();
        const auto series = X.cols();
        
        Matrix X_mean = Matrix(sample, series);
        for (int i=0; i<sample; i++){
          X_mean.row(i).setConstant(X.col(i).mean());
        }
        const Matrix X_center = X - X_mean;

        const auto eval = X_center.data();
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

  const ICA::Matrix mmm = ICA::Matrix::Random(3, 10);
  ICA::FastICA(mmm);
  return 0;
}