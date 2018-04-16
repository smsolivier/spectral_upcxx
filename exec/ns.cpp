#include "DataObjects.H"
#include "Writer.H"
#include <upcxx/upcxx.hpp> 
#include "Timer.H"

using namespace std; 

double gaussian(double x, double y, double xc, double yc, double width) {
	return exp(-pow(x - xc,2)/width)*exp(-pow(y-yc,2)/width); 
}

int main(int argc, char* argv[]) {
	upcxx::init(); 
	START_WTIMER(); 
	INT N = 16; 
	if (argc > 1) N = atoi(argv[1]); 
	array<INT,DIM> dims = {N, N, N}; 

	double T = 6; // end time 
	double K = .001; // time step 
	INT Nt = T/K; // number of time steps 
	double nu = 1; // viscosity 

	Writer writer("solution"); 

	Vector omega0(dims); 
	Vector v(dims, false); 
	writer.add(omega0, "omega"); 
	writer.add(v, "velocity"); 
	array<INT,DIM> start = omega0.getPStart(); 
	array<INT,DIM> end = omega0.getPEnd(); 
	array<INT,DIM> ind = {0,0,0}; 
	array<double,DIM> h; 
	double width = M_PI/8; // width of gaussian 
	double mag = 50; // multiplier on vorticity 
	double dist = M_PI/2; // separation between vorticies 
	for (int i=0; i<DIM; i++) {
		h[i] = 2.*M_PI/dims[i]; 
	}
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		double z = h[2]*ind[2]; 
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			double y = h[1]*ind[1]; 
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				double x = h[0]*ind[0]; 
				omega0[2].set(ind, mag*gaussian(x,y,M_PI-dist,M_PI,width) + 
					mag*gaussian(x,y,M_PI+dist,M_PI,width)); 
			}
		}
	}
	omega0.forward(); 

	Scalar psi(dims, false); 
	array<double,DIM> k; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				k = psi.freq(ind); 
				if (k[0] != 0 || k[1] != 0) {
					psi.set(ind, -1./(k[0]*k[0]+k[1]*k[1])*omega0[2][ind]); 
				}
			}
		}
	}
	Vector grad = psi.gradient(); 
	writer.add(grad, "grad"); 
	writer.add(psi, "psi"); 
	// v[0] = -1.*grad[1]; 
	-1.*grad[1]; 
	v[1] = grad[0]; 

	writer.write(); 
	upcxx::finalize();
}