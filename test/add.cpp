#include "Scalar.H"
#include <upcxx/upcxx.hpp> 
#include <iostream> 
#include "Timer.H"

using namespace std; 

int main(int argc, char* argv[]) {
	upcxx::init(); 
	INT mrank = upcxx::rank_me(); 
	INT N = 32; 
	if (argc > 1) N = atoi(argv[1]); 
	array<INT,DIM> dims = {N, N, N}; 
	Scalar a(dims); 
	a.memory(); 
	Scalar b(dims); 

	INT Nz = dims[2]/upcxx::rank_n(); 
	array<INT,DIM> ind = {0,0,0}; 
	Timer set("set"); 
	// for (ind[0]=0; ind[0]<dims[0]; ind[0]++) {
	// 	for (ind[1]=0; ind[1]<dims[1]; ind[1]++) {
	// 		for (ind[2]=mrank*Nz; ind[2]<(mrank+1)*Nz; ind[2]++) {
	// 			a.set(ind, 1.); 
	// 			b.set(ind, 2.); 
	// 		}
	// 	}
	// }
	for (INT i=0; i<a.sizePerProcessor(); i++) {
		a[i] = 1.; 
		b[i] = 2.; 
	}
	set.stop(); 

	Timer add("add"); 
	a.add(b); 
	add.stop();  

	bool wrong = false; 
	for (INT i=0; i<a.sizePerProcessor(); i++) {
		if (a[i].real() != 3) wrong = true; 
	}
	if (wrong) cout << "WRONG!" << endl; 
	upcxx::finalize(); 
}