#include <omp.h>
#include "DataObjects.H"
#include <upcxx/upcxx.hpp> 
#include <chrono> 
#include "CH_Timer.H"
#include "Writer.H"
#include <vector> 

#define CHECK

using namespace std; 

int main(int argc, char* argv[]) {
	upcxx::init(); 
#ifdef ZERO 
	if (upcxx::rank_me() == 0) 
		cout << "WARNING: FFT will be wrong if ZERO is defined" << endl; 
#endif
	INT N = 64; 
	int nruns = 10; 
	if (argc > 1) N = atoi(argv[1]); 
	if (argc > 2) nruns = atoi(argv[2]); 

	int mrank = upcxx::rank_me(); 

	array<INT,DIM> dims = {N, N, N}; 

	int nthreads = 0; 
#ifdef OMP 
	#pragma omp parallel 
	{
		nthreads = omp_get_num_threads(); 
	}
#endif

	Scalar s(dims); 
	Scalar ans(dims); 
	// s.memory(); 
	Writer writer; 
	writer.add(s, "s"); 

	for (int i=0; i<s.localSize(); i++) {
		s[i] = (double)rand()/RAND_MAX; 
		ans[i] = s[i]; 			
	}

	upcxx::barrier(); 

	chrono::time_point<chrono::system_clock> start; 
	chrono::duration<double> el; 
	double min = std::numeric_limits<double>::max(); 
	for (int i=0; i<nruns; i++) {
		start = chrono::system_clock::now(); 
		s.forward(); 
		s.inverse(); 		
		el = chrono::system_clock::now() - start; 
		if (el.count() < min) min = el.count(); 
	}
	upcxx::barrier(); 

	if (upcxx::rank_me() == 0) {
		cout << "n=" << upcxx::rank_n() << ", t=" << nthreads 
			<< ", min time = " << min << " seconds"; 
			#if defined PENCILS 
			cout << " (pencils)" << endl; 
			#elif defined SLABS 
			cout << " (slabs)" << endl; 
			#else 
			cout << endl; 
			#endif
	}

	// check if wrong 
#ifdef CHECK
	upcxx::global_ptr<bool> wrong_ptr = nullptr; 
	if (mrank == 0) wrong_ptr = upcxx::new_array<bool>(upcxx::rank_n()); 
	wrong_ptr = upcxx::broadcast(wrong_ptr, 0).wait(); 
	bool wrong = false; 
	for (int i=0; i<s.localSize(); i++) {
		if (abs(s[i] - ans[i]) > 1e-3) wrong = true;
	}
	upcxx::rput(wrong, wrong_ptr+mrank).wait(); 
	upcxx::barrier(); 
	if (mrank == 0) {
		bool* wrong_local = wrong_ptr.local(); 
		bool wrong_global = false; 
		for (int i=0; i<upcxx::rank_n(); i++) {
			if (wrong_local[i] == true) wrong_global = true; 
		}
		if (wrong_global) cout << "WRONG!" << endl; 
		else cout << "my man!" << endl; 
	}

	// compute error 
	for (int i=0; i<s.localSize(); i++) {
		s[i] = abs(ans[i] - s[i]); 
	}

	writer.write(); 
#endif

	CH_TIMER_REPORT(); 

	upcxx::finalize(); 
}