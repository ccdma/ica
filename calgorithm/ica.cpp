#define DEBUG_PROGLESS
// #define DEBUG_MATRIX
// #define EIGEN_USE_BLAS

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <sstream>
#include <Eigen/Dense>

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

#ifdef DEBUG_PROGLESS
	std::chrono::system_clock::time_point prev, now;
	prev = std::chrono::system_clock::now();
	now = std::chrono::system_clock::now();
	std::cout 
	<< "[PROGLESS] start fastica session"
	<< "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(now-prev).count() << std::endl;
	prev = now;
#endif

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

#ifdef DEBUG_PROGLESS
	now = std::chrono::system_clock::now();
	std::cout 
	<< "[PROGLESS] start fixed point method"
	<< "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(now-prev).count() << std::endl;
	prev = now;
#endif

#ifdef DEBUG_MATRIX
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

#ifdef DEBUG_MATRIX
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
			if (j==LOOP-1) printf("[WARN] loop limit exceeded\n");
		}

#ifdef DEBUG_PROGLESS
		now = std::chrono::system_clock::now();
		std::cout
		<< "[PROGLESS] end loop " << i
		<< "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(now-prev).count() << std::endl;
		prev = now;
#endif

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

	// チェビシェフ多項式の値を計算
	// オーバーフローを避けるため、漸化式を利用
	// n: 次数(n>=0) return: T_n(x)の値
	double EvalChebyt(const double x, const int n){
		double t0 = 1.0;
		double t1 = x;
		if (n == 0) {
			return t0;
		} else if (n == 1) {
			return t1;
		}
		double t2;
		for(int i=2; i<=n; i++){
			t2 = 2*x*t1-t0;
			t0 = t1;
			t1 = t2;
		}
		return t2;
	}
}

int main(){
	std::mt19937 mt(10);
	const auto sample = 20;
	const auto series = 100000;
	
	ICA::Matrix S(sample,series);
	#pragma omp parallel for
	for (int i=0; i<sample; i++){
		#pragma omp parallel for
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