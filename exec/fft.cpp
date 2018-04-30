#include "DataObjects.H"
#include <upcxx/upcxx.hpp> 
#include <chrono> 
#include "CH_Timer.H"
#include "Writer.H"

using namespace std; 

int main(int argc, char* argv[]) {
	upcxx::init(); 
#ifdef ZERO 
	cout << "WARNING: FFT will be wrong if ZERO is defined" << endl; 
#endif
	INT N = 64; 
	int nruns = 10; 
	if (argc > 1) N = atoi(argv[1]); 

	int mrank = upcxx::rank_me(); 

	array<INT,DIM> dims = {N, N, N}; 

	Vector v(dims); 
	Vector ans(dims); 
	v[0].memory(); 
	Writer writer; 
	writer.add(v, "V"); 

	for (int i=0; i<v.localSize(); i++) {
		for (int d=0; d<DIM; d++) {
			v[d][i] = (double)rand()/RAND_MAX; 
			ans[d][i] = v[d][i]; 			
		}
	}

	upcxx::barrier(); 

	for (int i=0; i<nruns; i++) {
		v.forward(); 
		v.inverse(); 		
	}

	upcxx::barrier(); 

	// check if wrong 
	upcxx::global_ptr<bool> wrong_ptr = nullptr; 
	if (mrank == 0) wrong_ptr = upcxx::new_array<bool>(upcxx::rank_n()); 
	wrong_ptr = upcxx::broadcast(wrong_ptr, 0).wait(); 
	bool wrong = false; 
	for (int i=0; i<v.localSize(); i++) {
		for (int d=0; d<DIM; d++) {
			if (abs(v[d][i] - ans[d][i]) > 1e-3) wrong = true; 			
		}
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
	for (int d=0; d<DIM; d++) {
		for (int i=0; i<v.localSize(); i++) {
			v[d][i] = abs(ans[d][i] - v[d][i]); 
		}
	}

	writer.write(); 

	CH_TIMER_REPORT(); 

	upcxx::finalize(); 
}