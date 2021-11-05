#define DEBUG_PROGLESS

#include "ica.cpp"

ICA::Matrix Pmatrix(ICA::Matrix& G){
	ICA::Matrix P = ICA::Matrix::Zero(G.rows(), G.cols());
	#pragma omp parallel for
	for(int i=0; i<G.rows(); i++){
		ICA::Vector row = G.row(i);
		ICA::Vector::Index maxId;
		row.cwiseAbs().maxCoeff(&maxId);
		const double x = row(maxId);
		P(i, maxId) = (x > 0) ? 1 : -1;
	}
	return P;
}

int main(){
	ICA::Reng reng(10);
	const auto sample = 3;
	const auto series = 1000;
	
	std::vector<ICA::Vector> s(sample);
	#pragma omp parallel for
	for (int i=0; i<sample; i++){
		s.at(i) = ICA::ChebytSeries(i+2, series, 0.2);
	}
	ICA::Matrix noncenterS = ICA::VStack(s);
	ICA::Matrix S = ICA::Centerize(noncenterS);

	const ICA::Matrix A = ICA::RandMatrix(sample, reng);
	ICA::Matrix X = A * S;
	auto result = ICA::FastICA(X);
	ICA::Matrix G = result.W * A;
	ICA::Matrix P = Pmatrix(G);

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