#include "Dist3D.H"
#include <upcxx/upcxx.hpp> 
#include <chrono> 

using namespace std; 

int main(int argc, char* argv[]) {
	upcxx::init(); 
	int N = 64; 
	int nruns = 1; 
	if (argc > 1) {
		N = atoi(argv[1]); 
	}

	array<int,DIM> dims = {N, N, N}; 
	if (upcxx::rank_me()==0) {
		cout << "memory = " << dims[0]*dims[1]*dims[2]*sizeof(complex<double>)/1e9
			<< " GB" << endl; 
	}

	Dist3D d(dims); 

	array<int,DIM> ind = {0,0,0}; 
	for (ind[0]=0; ind[0]<dims[0]; ind[0]++) {
		for (ind[1]=0; ind[1]<dims[1]; ind[1]++) {
			for (ind[2]=0; ind[2]<dims[2]; ind[2]++) {
				d.set(ind, 1); 
			}
		}
	}

	// speed test 
	double max = 0; 
	for (int i=0; i<nruns; i++) {
		chrono::time_point<chrono::system_clock> start = chrono::system_clock::now(); 
		d.forward(); 
		d.inverse(); 
		chrono::duration<double> el = chrono::system_clock::now() - start; 
		if (el.count() > max) max = el.count(); 
	}

	cout << upcxx::rank_me() << ", max time = " << max << endl; 

	upcxx::barrier(); 
	bool wrong = false; 
	for (ind[0]=0; ind[0]<dims[0]; ind[0]++) {
		for (ind[1]=0; ind[1]<dims[1]; ind[1]++) {
			for (ind[2]=0; ind[2]<dims[2]; ind[2]++) {
				if (d[ind].real() != 1) {
					wrong = true; 
				}
			}
		}
	}
	if (wrong) cout << "WRONG!" << endl; 
	upcxx::finalize(); 
}