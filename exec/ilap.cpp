#include "DataObjects.H"
#include <upcxx/upcxx.hpp> 
#include <chrono> 
#include <omp.h>

using namespace std; 

int main(int argc, char* argv[]) {
	upcxx::init(); 
	INT N = 32; 
	int nruns = 15; 
	if (argc > 1) N = atoi(argv[1]); 
	array<INT,DIM> dims = {N, N, N}; 

	int nthreads = 0; 
	#pragma omp parallel 
	{
		nthreads = omp_get_num_threads(); 
	}

	Scalar s(dims, false); 

	chrono::time_point<chrono::system_clock> start; 
	chrono::duration<double> el; 
	double min = numeric_limits<double>::max(); 
	for (int i=0; i<nruns; i++) {
		start = chrono::system_clock::now(); 
		s.laplacian_inverse(); 
		upcxx::barrier(); 
		el = chrono::system_clock::now() - start; 
		if (el.count() < min) min = el.count(); 
	}

	if (upcxx::rank_me() == 0) {
		cout << "n=" << upcxx::rank_n() << ", t=" << nthreads 
			<< ", min time = " << min << " seconds" << endl; 
	}

	upcxx::finalize();
}