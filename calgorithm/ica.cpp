#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <sstream>
#include <eigen3/Eigen/Dense>

#define DEBUG
// #define EIGEN_USE_BLAS

namespace ICA {

    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    const int LOOP = 100;

    Matrix RandMatrix(int size, std::mt19937& engine){
      std::uniform_real_distribution<double> distribution(-0.5, 0.5);
      auto generator = [&] (double dummy) {return distribution(engine);};
      return Matrix::Zero(size, size).unaryExpr(generator);
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
        std::mt19937 mt(0);
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

#ifdef DEBUG
        // 単位行列であることを確認
        std::cout << (X_whiten * X_whiten.transpose()) / double(X_whiten.cols() - 1) << std::endl;
#endif

        const auto g = [](double bx) { return std::pow(bx, 3); };
        const auto g2 = [](double bx) { return 3*std::pow(bx, 2); };

        const auto I = X_whiten.rows();
        auto B = RandMatrix(I, mt);

        for(int i=0; i<I; i++){
          Normalize(B, i);
        }

#ifdef DEBUG
        // 単位行列であることを確認
        std::cout << B * B.transpose() << std::endl;
#endif

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
            if (j==LOOP-1) printf("[WARN] loop limit exceeded");
          }
        }
        Matrix Y = B.transpose() * X_whiten;

        return FastICAResult{Y};
    };
    
    std::vector<double> ToStdVec(Vector& v1){
      std::vector<double> v2(v1.data(), v1.data() + v1.size());
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
  std::mt19937 mt(10);
  const auto sample = 3;
  const auto series = 100;
  
  ICA::Matrix S(sample,series);
  for (int i=0; i<sample; i++){
    for (int j=0;j<series;j++){
      S(i,j) = std::sin((i+2)*(double)j*0.02);
    }
  }
  const ICA::Matrix A = ICA::RandMatrix(sample, mt);
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