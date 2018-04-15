#include "Scalar.H"
#include <upcxx/upcxx.hpp>
#include <iostream>
#include <fstream> 
#include <chrono> 
#include "VisitWriter.H"

using namespace std;  

int main() {
	upcxx::init(); 
	int N = 128; 
	array<int,DIM> dims = {N, N, N}; 
	if (upcxx::rank_me() == 0) {
		cout << "memory = " << dims[0]*dims[1]*dims[2]*sizeof(double)/1e9 << endl; 
	}

	Scalar d(dims); 

	int mrank = upcxx::rank_me(); 

	int Nz = ceil((double)dims[2]/upcxx::rank_n()); 

	double xb = 2*M_PI; 
	float hx = xb/(dims[0]); 
	float hy = xb/(dims[1]); 
	float hz = xb/(dims[2]); 

	double s = 0; 
	double s2 = 0; 		
	bool wrong = false; 
	wrong = false; 
	array<int,DIM> ind = {0,0,0}; 
	chrono::time_point<chrono::system_clock> start = chrono::system_clock::now(); 
	for (ind[0]=0; ind[0]<dims[0]; ind[0]++) {
		for (ind[1]=0; ind[1]<dims[1]; ind[1]++) {
			for (ind[2]=mrank*Nz; ind[2]<(mrank+1)*Nz and ind[2]<dims[2]; ind[2]++) {
				// if (d[ind] != n) {
				// 	wrong = true;  
				// }
				d.set(ind, sin(ind[0]*hx)*sin(ind[1]*hy)); 
				// d.set(ind, sin(ind[2]*hx)); 
				// d.set(ind, 1.); 
			}
		}
	}
	chrono::duration<double> el = chrono::system_clock::now() - start; 
	cout << "n = " << mrank << ": " << el.count() << " seconds" << endl; 
	if (wrong) {cout << "wrong" << endl; }

	upcxx::barrier(); 

	d.forward(); 
	d.inverse(); 

	vector<float> x(dims[0]); 
	vector<float> y(dims[1]); 
	vector<float> z(dims[2]);
	for (int i=0; i<dims[0]; i++) {
		x[i] = i*hx; 
	} 
	for (int i=0; i<dims[1]; i++) {
		y[i] = i*hy; 
	}
	for (int i=0; i<Nz; i++) {
		z[i] = (mrank*Nz + i)*hz; 
	}

	// ouput your data 
	int nvars = 1; 
	float* vals = new float[dims[0]*dims[1]*Nz]; 
	cdouble* local = d.getLocal(); 
	for (int i=0; i<dims[0]*dims[1]*Nz; i++) {
		vals[i] = local[i].real(); 
	}
	int vardim[nvars]; 
	vardim[0] = 1; 
	int centering[nvars]; 
	centering[0] = 1; 
	const char* varnames[nvars]; 
	varnames[0] = "u"; 

	string fname = "solution" + to_string(mrank); 
	array<int,DIM> tdims = {dims[0], dims[1], Nz}; 
	write_rectilinear_mesh(fname.c_str(), 1, &tdims[0], &x[0], &y[0], &z[0],
		nvars, &vardim[0], &centering[0], varnames, &vals); 

	// write master file 
	if (mrank==0) {
		ofstream out("solution.visit"); 
		out << "!NBLOCKS " << upcxx::rank_n() << endl; 
		for (int i=0; i<upcxx::rank_n(); i++) {
			out << "solution" << i << ".vtk" << endl; 
		}
		out.close(); 
	}
	upcxx::finalize(); 
}