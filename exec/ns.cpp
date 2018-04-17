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
	INT N = 32; 
	if (argc > 1) N = atoi(argv[1]); 
	array<INT,DIM> dims = {N, N, N}; 

	double T = 6; // end time 
	double K = .001; // time step 
	INT Nt = T/K; // number of time steps 
	double nu = .001; // viscosity 

	// create a writer 
	Writer writer("solution"); 
	writer.setFreq(20); 

	// initialize variables 
	// vorticities 
	Vector omega0(dims); 
	Vector omega1(dims, false); 
	Vector omega(dims, false); 
	// velocities 
	Vector V0(dims, false); 
	Vector V1(dims, false); 
	Vector V(dims, false); 
	Vector Vhalf(dims, false); // fractional half step 

	// pressure 
	Scalar Pi0(dims, false); 
	Scalar Pi1(dims, false); 
	Scalar Pi(dims, false); 

	// divergence 
	Scalar div(dims, false); 

	// track variables in writer 
	writer.add(omega, "omega"); 
	writer.add(V, "velocity"); 
	writer.add(Pi, "Pi"); 
	writer.add(div, "divergence"); 

	upcxx::barrier(); 
	div.memory(); 

	// start indices 
	array<INT,DIM> start = omega0.getPStart(); 
	// end indices 
	array<INT,DIM> end = omega0.getPEnd(); 
	// index into arrays 
	array<INT,DIM> ind = {0,0,0}; 
	// cell spacing 
	array<double,DIM> h; 
	double width = M_PI/8; // width of gaussian 
	double mag = 50; // multiplier on vorticity 
	double dist = M_PI/2; // separation between vorticies 
	for (int i=0; i<DIM; i++) {
		h[i] = 2.*M_PI/dims[i]; 
	}

	// set initial vorticity 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		double z = h[2]*ind[2]; 
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			double y = h[1]*ind[1]; 
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				double x = h[0]*ind[0]; 
				omega0[2][ind] = mag*gaussian(x,y,M_PI-dist,M_PI,width) + 
					mag*gaussian(x,y,M_PI+dist,M_PI,width); 
			}
		}
	}
	omega0.forward(); 

	// compute velocity field 
	Scalar psi(dims, false); 
	array<double,DIM> k; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				k = psi.freq(ind); 
				if (k[0] != 0 || k[1] != 0) {
					psi[ind] = -1./(k[0]*k[0]+k[1]*k[1])*omega0[2][ind]; 
				}
			}
		}
	}
	Vector grad = psi.gradient(); 
	V0[0] = -1.*grad[1]; 
	V0[1] = grad[0]; 

	// Pi0 
	Pi0 = (V0.cross(omega0) + nu*V0.laplacian()).divergence(); 
	Pi0.laplacian_inverse(); 

	// first time step with Forward Euler 
	V1 = V0 + K*(V0.cross(omega0) + nu*V0.laplacian() - Pi0.gradient()); 

	// compute omega1 = curl(V1) 
	omega1 = V1.curl(); 

	// Pi1 
	Pi1 = (V1.cross(omega1) + nu*V1.laplacian()).divergence(); 
	Pi1.laplacian_inverse(); 

	// do time stepping 
	for (int t=1; t<Nt+1; t++) {
		// AB2 step 
		Vhalf = V1 + K/2*(3.*V1.cross(omega1) - V0.cross(omega0)) 
			- K/2*(3.*Pi1.gradient() - Pi0.gradient()); 

		// CN step 
		V = Vhalf + nu*K/2*V1.laplacian(); 
		for (int d=0; d<DIM; d++) {
			V[d].laplacian_inverse(1, -nu*K/2); 
		}

		// compute vorticity 
		omega = V.curl(); 

		// compute Pi 
		Pi = (V.cross(omega) + nu*V.laplacian()).divergence(); 
		Pi.laplacian_inverse(); 

		// compute divergence 
		div = V.divergence(); 

		// save histories 
		V0 = V1; 
		V1 = V; 
		omega0 = omega1; 
		omega1 = omega; 
		Pi0 = Pi1; 
		Pi1 = Pi; 

		// write to VTK 
		writer.write(); 

		cout << t*K/T << "\r"; 
		cout.flush(); 
	}

	upcxx::finalize();
}