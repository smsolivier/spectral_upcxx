#include "Scalar.H"
#include "VisitWriter.H"
#include <upcxx/upcxx.hpp> 
#include <chrono> 
#include <unistd.h>
#include <fstream>
#include "Timer.H"

using namespace std; 

int main(int argc, char* argv[]) {
	upcxx::init(); 
	INT N = 64; 
	INT nruns = 1; 
	if (argc > 1) {
		N = atoi(argv[1]); 
	}	

	INT mrank = upcxx::rank_me(); 

	array<INT,DIM> dims = {N, N, N}; 

	Scalar d(dims); 
	Scalar ans(dims); 
	d.memory(); 

	INT Nz = dims[2]/upcxx::rank_n(); 
	array<INT,DIM> ind = {0,0,0}; 

	double hx = 2.*M_PI/N; 
	double hy = hx; 
	double hz = hx; 
	srand(upcxx::rank_me()); 
	Timer set("set"); 
	// for (ind[0]=0; ind[0]<dims[0]; ind[0]++) {
	// 	for (ind[1]=0; ind[1]<dims[1]; ind[1]++) {
	// 		for (ind[2]=mrank*Nz; ind[2]<(mrank+1)*Nz; ind[2]++) {
	// 			double val = (double)rand()/RAND_MAX; 
	// 			// d.set(ind, 1); 
	// 			d.set(ind, val); 
	// 			ans.set(ind, val); 
	// 		}
	// 	}
	// }
	for (INT i=0; i<d.sizePerProcessor(); i++) {
		d[i] = (double)rand()/RAND_MAX; 
		ans[i] = d[i]; 
	}
	set.stop(); 

	// speed test 
	double max = 0; 
	for (int i=0; i<nruns; i++) {
		Timer time("transform"+to_string(i)); 
		d.forward(); 
		d.inverse(); 
		time.stop(); 
	}
	upcxx::barrier(); 

	// check if wrong 
	upcxx::global_ptr<bool> wrong_ptr = nullptr; 
	if (mrank == 0) wrong_ptr = upcxx::new_array<bool>(upcxx::rank_n()); 
	wrong_ptr = upcxx::broadcast(wrong_ptr, 0).wait(); 
	bool wrong = false; 
	for (int i=0; i<d.sizePerProcessor(); i++) {
		if (abs(d[i] - ans[i]) > 1e-3) wrong = true; 
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

	// write to file 
	// vector<float> x(dims[0]); 
	// vector<float> y(dims[1]); 
	// vector<float> z(Nz);
	// for (INT i=0; i<dims[0]; i++) {
	// 	x[i] = i*hx; 
	// } 
	// for (INT i=0; i<dims[1]; i++) {
	// 	y[i] = i*hy; 
	// }
	// for (INT i=0; i<Nz; i++) {
	// 	z[i] = (mrank*Nz + i)*hz; 
	// }

	// // ouput your data 
	// int nvars = 1; 
	// float* vals = new float[dims[0]*dims[1]*Nz]; 
	// cdouble* local = d.getLocal(); 
	// cdouble* ans_local = ans.getLocal(); 
	// for (INT i=0; i<dims[0]*dims[1]*Nz; i++) {
	// 	vals[i] = abs(local[i].real() - ans_local[i].real()); 
	// 	// vals[i] = local[i].real(); 
	// }
	// int vardim[nvars]; 
	// vardim[0] = 1; 
	// int centering[nvars]; 
	// centering[0] = 1; 
	// const char* varnames[nvars]; 
	// varnames[0] = "u"; 

	// string fname = "solution" + to_string(mrank); 
	// array<int,DIM> tdims = {(int)dims[0], (int)dims[1], (int)Nz}; 
	// write_rectilinear_mesh(fname.c_str(), 1, &tdims[0], &x[0], &y[0], &z[0],
	// 	nvars, &vardim[0], &centering[0], varnames, &vals); 

	// // write master file 
	// if (mrank==0) {
	// 	ofstream out("solution.visit"); 
	// 	out << "!NBLOCKS " << upcxx::rank_n() << endl; 
	// 	for (int i=0; i<upcxx::rank_n(); i++) {
	// 		out << "solution" << i << ".vtk" << endl; 
	// 	}
	// 	out.close(); 
	// }
	PRINT_TIMER_REPORT(upcxx::rank_me()); 
	upcxx::finalize(); 
}