#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <sstream>

// #define EIGEN_USE_BLAS

#include <eigen3/Eigen/Dense>
using std::vector;
using std::srand;


#define DEBUG false
#define LOOP 1000

namespace ICA {

    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    Matrix RandMatrix(int size){
      return Matrix::Random(size, size);
    };

    // 正方行列でなくてはいけない
    // i番目を直行空間に射影
    void Normalize(Matrix& mat, int i){
      const auto size = mat.cols();
      if (i>0){
        mat.col(i) = mat.col(i) - mat.block(0, 0, size, i) * mat.block(0, 0, size, i).transpose() * mat.col(i);
      }
      mat.col(i) = mat.col(i) / std::sqrt(mat.col(i).squaredNorm());
    }

    struct FastICAResult {
        Matrix Y;
    };

    FastICAResult FastICA(const Matrix& X) {
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
        auto B = RandMatrix(I);

        for(int i=0; i<I; i++){
          Normalize(B, i);
        }

        if(DEBUG) std::cout << B * B.transpose() << std::endl;

        for(int i=0; i<I; i++){
          for(int j=0; j<LOOP; j++){
            // 値のコピー
            const Vector prevBi = B.col(i);

            const auto collen = X_whiten.cols();
            Matrix ave(I, collen);
            #pragma omp parallel for
            for(int k=0; k<collen; k++){
              const Vector x = X_whiten.col(k);
              ave.col(k) = g(x.dot(B.col(i)))*x - g2(x.dot(B.col(i)))*B.col(i);  
            }
            B.col(i) = ave.rowwise().mean();
            Normalize(B, i);
            const auto diff = std::abs(prevBi.dot(B.col(i)));
            if (1.0 - 1.e-8 < diff && diff < 1.0 + 1.e-8) break;
            if (j==LOOP) printf("[WARN] loop limit exceeded");
          }
        }
        Matrix Y = B.transpose() * X_whiten;

        return FastICAResult{Y};
    };
    
    std::vector<double> ToStdVec(Vector& v1){
      vector<double> v2(v1.data(), v1.data() + v1.size());
      return v2;
    }

    void WriteMatrix(std::stringstream& ss, Matrix& mat){
      const auto sample = mat.rows();
      const auto series = mat.cols();
      for (int i=0; i<sample; i++){
        for (int j=0;j<series;j++){
          ss << mat(i,j) << ",";
        }
        ss << std::endl;
      }
    }
}

int main(){
  srand(0);
  const auto sample = 4;
  const auto series = 100000;
  ICA::Matrix S(sample,series);
  for (int i=0; i<sample; i++){
    for (int j=0;j<series;j++){
      S(i,j) = std::sin((i+2)*(double)j*0.02);
    }
  }
  const ICA::Matrix A = ICA::Matrix::Random(sample, sample);
  ICA::Matrix X = A * S;
  auto result = ICA::FastICA(X);

  std::stringstream ss;
  ICA::WriteMatrix(ss, S);
  ss << std::endl;
  ICA::WriteMatrix(ss, X);
  ss << std::endl;
  ICA::WriteMatrix(ss, result.Y);

  std::ofstream outputfile("test.csv");
  outputfile << ss.rdbuf();
  outputfile.close();
  return 0;
}