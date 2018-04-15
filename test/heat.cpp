#include "Scalar.H"
#include "VisitWriter.H"
#include <upcxx/upcxx.hpp> 
#include "Timer.H"

using namespace std; 

int main(int argc, char* argv[]) {
	upcxx::init(); 
	START_WTIMER();
	int N = 32; 
	if (argc > 1) N = atoi(argv[1]); 

	int mrank = upcxx::rank_me(); 

	array<INT,DIM> dims = {N, N, N}; 

	Scalar u(dims); 
	Scalar u0(dims); 
	u.memory(); 

	INT Nz = dims[2]/upcxx::rank_n(); 

	double hx = 2.*M_PI/N; 
	double hy = hx; 
	double hz = hx; 
	array<INT,DIM> ind = {0,0,0}; 
	Timer timer_set("set"); 
	for (ind[2]=mrank*Nz; ind[2]<(mrank+1)*Nz; ind[2]++) {
		double z = hz*ind[2]; 
		for (ind[1]=0; ind[1]<dims[1]; ind[1]++) {
			double y = hy*ind[1]; 
			for (ind[0]=0; ind[0]<dims[0]; ind[0]++) {
				double x = hx*ind[0]; 
				u0.set(ind, exp(-pow(x-M_PI, 2)/.01)
					*exp(-pow(y-M_PI,2)/.01)
					*exp(-pow(z-M_PI,2)/.01)); 
			}
		}
	}
	timer_set.stop(); 

	u0.forward(); 

	int Nt = 10; 
	double Tend = 5; 
	for (int t=1; t<Nt+1; t++) {
		double T = (double)t*Tend/Nt; 

		for (ind[2]=mrank*Nz; ind[2]<(mrank+1)*Nz; ind[2]++) {
			for (ind[1]=0; ind[1]<dims[1]; ind[1]++) {
				for (ind[0]=0; ind[0]<dims[0]; ind[0]++) {
					array<int,DIM> k = u.freq(ind); 
					u.set(ind, 
						u0[ind]*exp(-(k[0]*k[0] + k[1]*k[1] + k[2]*k[2])*T)); 
				}
			}
		}

		u.inverse(); 

		Timer timer_write("write to file"); 
		vector<float> x(dims[0]); 
		vector<float> y(dims[1]); 
		vector<float> z(Nz);
		for (INT i=0; i<dims[0]; i++) {
			x[i] = i*hx; 
		} 
		for (INT i=0; i<dims[1]; i++) {
			y[i] = i*hy; 
		}
		for (INT i=0; i<Nz; i++) {
			z[i] = (mrank*Nz + i)*hz; 
		}

		// ouput your data 
		int nvars = 1; 
		float* vals = new float[dims[0]*dims[1]*Nz]; 
		cdouble* local = u.getLocal(); 
		for (INT i=0; i<dims[0]*dims[1]*Nz; i++) {
			vals[i] = local[i].real(); 
		}
		int vardim[nvars]; 
		vardim[0] = 1; 
		int centering[nvars]; 
		centering[0] = 1; 
		const char* varnames[nvars]; 
		varnames[0] = "u"; 

		string fname = "solution" + to_string(mrank) + "_" + to_string(t); 
		array<int,DIM> tdims = {(int)dims[0], (int)dims[1], (int)Nz}; 
		write_rectilinear_mesh(fname.c_str(), 1, &tdims[0], &x[0], &y[0], &z[0],
			nvars, &vardim[0], &centering[0], varnames, &vals); 

		delete[] vals; 
		timer_write.stop(); 

		if (upcxx::rank_me() == 0) cout << t << endl; 
	}

	// write master file 
	if (mrank==0) {
		ofstream out("solution.visit"); 
		out << "!NBLOCKS " << upcxx::rank_n() << endl; 
		for (int t=1; t<Nt+1; t++) {
			for (int i=0; i<upcxx::rank_n(); i++) {
				out << "solution" << i << "_" << t << ".vtk" << endl; 
			}			
		}
		out.close(); 
	}
	upcxx::finalize();  
}