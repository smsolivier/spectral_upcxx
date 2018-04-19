#include "DataObjects.H"
#include "Writer.H"
#include <upcxx/upcxx.hpp> 

using namespace std; 

int main(int argc, char* argv[]) {
	upcxx::init(); 
	INT N = 32; 
	if (argc > 1) N = atoi(argv[1]); 
	array<INT,DIM> dims = {N, N, N}; 

	Scalar u(dims, false); 
	Scalar u0(dims, true); 
	u.memory(); 

	double h = 2*M_PI/N; 
	array<INT,DIM> start = u.getPStart(); 
	array<INT,DIM> end = u.getPEnd(); 
	array<INT,DIM> ind = {0,0,0}; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		double z = h*ind[2]; 
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			double y = h*ind[1]; 
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				double x = h*ind[0]; 
				u0[ind] = exp(-pow(x-M_PI,2)/.01)*exp(-pow(y-M_PI,2)/.01)*
					exp(-pow(z-M_PI,2)/.01); 

			}
		}
	}

	Writer writer; 
	writer.add(u, "u"); 
	writer.setFreq(100);

	u0.forward(); 

	int Nt = 2000; 
	double Tend = 5; 
	double K = Tend/Nt; 
	for (int t=1; t<Nt+1; t++) {
		u = u0 + K/2*u0.laplacian(); 
		u.laplacian_inverse(1, -K/2); 

		u0 = u; 

		cout << t*K/Tend << "\r"; 
		cout.flush(); 

		// writer.write(); 
	}

	upcxx::finalize(); 
}