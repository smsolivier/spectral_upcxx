#include <complex> 
#include <chrono> 
#include <array>
#include <iostream>

using namespace std; 

typedef complex<double> cdouble; 
typedef uint64_t INT; 

array<double,DIM> freq(array<INT,DIM> ind, array<INT,DIM> m_dims) { 
	array<double,DIM> k; 
	for (int i=0; i<DIM; i++) {
		if (ind[i] <= m_dims[i]/2) k[i] = (double)ind[i]; 
		else k[i] = -1.*(double)m_dims[i] + (double)ind[i];
	}

	return k; 
}

void ilap(cdouble* f, array<INT,DIM> dims) {
	double a = 0; 
	double b = 1;
	array<INT,DIM> ind = {0,0,0}; 
	array<double,DIM> K; 
	for (ind[2]=0; ind[2]<dims[2]; ind[2]++) {
		for (ind[1]=0; ind[1]<dims[1]; ind[1]++) {
			for (ind[0]=0; ind[0]<dims[0]; ind[0]++) {
				K = freq(ind, dims); 
				double sum = 0; 
				for (int d=0; d<DIM; d++) {
					sum += K[d]*K[d]; 
				}
				if (sum != 0) {
					f[ind[0]+ind[1]*dims[0]+ind[2]*dims[0]*dims[1]] /= (a - b*sum); 
				}
			}
		}
	}
}

int main(int argc, char* argv[]) {
	INT N = 32; 
	if (argc > 1) N = atoi(argv[1]); 
	int nruns = 15; 
	array<INT,DIM> dims = {N,N,N}; 

	cdouble* f = new cdouble[N*N*N]; 

	chrono::time_point<chrono::system_clock> start; 
	chrono::duration<double> el; 
	double min = numeric_limits<double>::max(); 
	for (int i=0; i<nruns; i++) {
		start = chrono::system_clock::now(); 
		ilap(f, dims); 		
		el = chrono::system_clock::now() - start; 
		if (el.count() < min) min = el.count(); 
	}

	cout << "min time = " << min << endl; 
}