#include "DataObjects.H"
#include <upcxx/upcxx.hpp> 
#include <chrono> 
#include "CH_Timer.H"

using namespace std; 

int main(int argc, char* argv[]) {
	upcxx::init(); 
	INT N = 64; 
	int nruns = 1; 
	if (argc > 1) N = atoi(argv[1]); 

	int mrank = upcxx::rank_me(); 

	array<INT,DIM> dims = {N, N, N}; 

	Scalar s(dims); 
	Scalar ans(dims); 
	s.memory(); 

	double min = 100000.;  
	for (int i=0; i<s.localSize(); i++) {
		s[i] = (double)rand()/RAND_MAX; 
		ans[i] = s[i]; 
	}

	upcxx::barrier(); 

	s.forward(); 
	s.inverse(); 

	upcxx::barrier(); 

	// check if wrong 
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

	CH_TIMER_REPORT(); 

	upcxx::finalize(); 
}