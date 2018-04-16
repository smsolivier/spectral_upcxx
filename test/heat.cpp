#include "DataObjects.H"
#include "Writer.H"
#include <upcxx/upcxx.hpp> 
#include "Timer.H"

using namespace std; 

int main(int argc, char* argv[]) {
	upcxx::init(); 
	START_WTIMER();
	INT N = 32; 
	if (argc > 1) N = atoi(argv[1]); 

	int mrank = upcxx::rank_me(); 

	array<INT,DIM> dims = {N, N, N}; 

	Scalar u(dims, false); 
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

	Writer writer("solution"); 
	writer.add(u, "u"); 

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

		writer.write(); 

		if (upcxx::rank_me() == 0) cout << t << endl; 
	}

	upcxx::finalize();  
}