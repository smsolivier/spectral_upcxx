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

	int mrank = upcxx::rank_me(); 

	array<int,DIM> dims = {N, N, N}; 
	if (upcxx::rank_me()==0) {
		cout << "memory = " << dims[0]*dims[1]*dims[2]*sizeof(double)*2/1e9
			<< " GB" << endl; 
	}

	Dist3D d(dims); 

	int Nz = dims[2]/upcxx::rank_n(); 
	chrono::duration<double> el; 
	chrono::time_point<chrono::system_clock> start = chrono::system_clock::now(); 
	array<int,DIM> ind = {0,0,0}; 
	for (ind[0]=0; ind[0]<dims[0]; ind[0]++) {
		for (ind[1]=0; ind[1]<dims[1]; ind[1]++) {
			for (ind[2]=mrank*Nz; ind[2]<(mrank+1)*Nz; ind[2]++) {
				d.set(ind, 1); 
			}
		}
	}
	upcxx::barrier(); 
	if (upcxx::rank_me() == 0) {
		el = chrono::system_clock::now() - start; 
		cout << "set time = " << el.count() << endl; 
	}

	// speed test 
	double max = 0; 
	for (int i=0; i<nruns; i++) {
		start = chrono::system_clock::now(); 
		d.forward(); 
		d.inverse(); 
		el = chrono::system_clock::now() - start; 
		if (el.count() > max) max = el.count(); 
	}

	cout << upcxx::rank_me() << ", max time = " << max << endl; 

	upcxx::barrier(); 
	bool wrong = false; 
	for (ind[0]=0; ind[0]<dims[0]; ind[0]++) {
		for (ind[1]=0; ind[1]<dims[1]; ind[1]++) {
			for (ind[2]=mrank*Nz; ind[2]<(mrank+1)*Nz; ind[2]++) {
				if (d[ind].real() != 1) {
					wrong = true; 
				}
			}
		}
	}
	if (wrong) cout << "WRONG!" << endl; 
	upcxx::finalize(); 
}