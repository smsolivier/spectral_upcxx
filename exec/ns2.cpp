#include "DataObjects.H"
#include "Writer.H"
#include <upcxx/upcxx.hpp> 
#include "CH_Timer.H"
#include "Particle.H"
#include <fstream> 
#include <chrono> 
#include <omp.h>

using namespace std; 

double gaussian(double x, double y, double xc, double yc, double width) {
	return exp(-pow(x - xc,2)/width)*exp(-pow(y-yc,2)/width); 
}

void writeParticles(vector<Particle>& parts, int t) {
	ofstream out("p"+to_string(t)+".3D"); 
	out << "x y z E" << endl; 
	for (int i=0; i<parts.size(); i++) {
		array<double,DIM> pos = parts[i].position(); 
		out << pos[0] << " " << pos[1] << " " 
			<< pos[2] << " " << parts[i].energy() << endl; 
	}
	out.close(); 
}

int main(int argc, char* argv[]) {
	upcxx::init(); 
#ifndef ZERO 
	cout << "WARNING: unstable if ZERO is not defined" << endl; 
#endif
	int nthreads = 1; 
#ifdef OMP 
	#pragma omp parallel 
	{
		nthreads = omp_get_num_threads(); 
	}
#endif
	INT N = 16; 
	if (argc > 1) N = atoi(argv[1]); 
	array<INT,DIM> dims = {N, N, N}; 

	double T = 12; // end time 
	double K = .001; // time step 
	INT Nt = T/K; // number of time steps 
	double nu = .001; // viscosity 
	int NSAVES = 300; 
	int mod = Nt/NSAVES; 

	// create a writer 
	Writer writer; 
	writer.setFreq(mod); 

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
	Scalar G(dims, false); // AB2 combination of Pi0 and Pi1 

	// divergence 
	Scalar div(dims, false); 

	// track variables in writer 
	writer.add(omega, "vorticity"); 
	writer.add(V, "velocity"); 
	writer.add(G, "G"); 
	writer.add(div, "div"); 

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
				// omega0[2][ind] = mag*gaussian(x,y,M_PI,M_PI, width); 
				// omega0[2][ind] = mag*gaussian(x,y,M_PI-dist,M_PI-dist,width) + 
					// mag*gaussian(x,y,M_PI+dist,M_PI-dist,width) + 
					// mag*gaussian(x,y,M_PI,M_PI+dist,width); 
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

	// compute G = dt/2*(3Pi1 - Pi0) 
	G = K/2*(3*Pi1 - Pi0); 

	// store AB2 cross product combination 
	Vector F; 

	// store previous time steps cross product 
	Vector cross0 = V0.cross(omega0); 
	Vector cross1; 
	Vector lap1; 

	#ifdef PARTICLES
	int Np = 10000; 
	vector<Particle> parts(Np); 
	#endif

	upcxx::barrier(); 
	chrono::time_point<chrono::system_clock> timer = chrono::system_clock::now(); 

	int n = 0; 
	// do time stepping 
	for (int t=1; t<Nt+1; t++) {

		// compute cross products 
		cross1 = V1.cross(omega1); 
		F = K/2*(3.*cross1 - cross0); 

		// compute laplacian of V1 (its used twice) 
		lap1 = nu*K*V1.laplacian(); 

		// compute G 
		G = (V1 + lap1 + F).divergence(); 
		G.laplacian_inverse(); 

		// AB2 step 
		Vhalf = V1 + F - G.gradient();

		// CN step 
		V = Vhalf; 
		for (int d=0; d<DIM; d++) {
			V[d].laplacian_inverse(1, -nu*K); 
		}

		#ifdef PARTICLES
		Vector vtmp; 
		V.inverse(vtmp); 
		{
			CH_TIMERS("move particles"); 
			#pragma omp parallel for 
			for (int i=0; i<Np; i++) {
				upcxx::default_persona_scope();
				parts[i].move(vtmp, K); 
			}
			if ((t-1) % mod == 0) {
				writeParticles(parts, n++); 	
			}
		}
		#endif

		// compute vorticity 
		omega = V.curl(); 

		// compute divergence 
		div = V.divergence(); 
		if (div.average() > 1e-12) cout << "avg div = " << div.average() << endl; 

		// save histories 
		V0 = V1; 
		V1 = V; 
		omega0 = omega1; 
		omega1 = omega; 
		cross0 = cross1; 

		// write to VTK 
		writer.write(); 

		if (upcxx::rank_me() == 0) {
			cout << K*t/T << "\r"; 
			cout.flush(); 
		}
	}

	CH_TIMER_REPORT();
	chrono::duration<double> el = chrono::system_clock::now() - timer; 
	if (upcxx::rank_me() == 0) {
		cout << "n=" << upcxx::rank_n() << ", t=" << nthreads 
			<< ", Wtime = " << el.count() << endl; 
	}
	upcxx::finalize();
}