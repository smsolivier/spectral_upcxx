#include "Particle.H"
#include "CH_Timer.H"
#include <iostream> 

Particle::Particle() {
	// set to random location in [0,2pi] 
	for (int i=0; i<DIM; i++) {
		m_loc[i] = (double)rand()/RAND_MAX*2*M_PI; 
	}
}

void Particle::move(const Vector& velocity, double K) {
	array<INT,DIM> ind; 
	array<INT,DIM> dims = velocity.getDims(); 
	array<double,DIM> h; 
	array<double,DIM> rd; 
	for (int d=0; d<DIM; d++) {
		h[d] = 2*M_PI/dims[d]; 
		ind[d] = m_loc[d]/h[d]; 
		if (ind[d]+1 >= dims[d]) ind[d] = dims[d]-2; 
		rd[d] = (m_loc[d] - ind[d]*h[d])/h[d]; 
	}

	// linear interpolation 
	for (int d=0; d<DIM; d++) {
		m_v[d] = velocity[d][ind].real()*(1-rd[0])*(1-rd[1])*(1-rd[2]) +
			velocity[d][{ind[0]+1,ind[1],ind[2]}].real()*rd[0]*(1-rd[1])*(1-rd[2]) +
			velocity[d][{ind[0]+1,ind[1]+1,ind[2]}].real()*rd[0]*rd[1]*(1-rd[2]) +
			velocity[d][{ind[0],ind[1]+1,ind[2]}].real()*(1-rd[0])*rd[1]*(1-rd[2]) + 
			velocity[d][{ind[0],ind[1],ind[2]+1}].real()*(1-rd[0])*(1-rd[1])*(rd[2]) +
			velocity[d][{ind[0]+1,ind[1],ind[2]+1}].real()*rd[0]*(1-rd[1])*(rd[2]) +
			velocity[d][{ind[0]+1,ind[1]+1,ind[2]+1}].real()*rd[0]*rd[1]*(rd[2]) +
			velocity[d][{ind[0],ind[1]+1,ind[2]+1}].real()*(1-rd[0])*rd[1]*(rd[2]); 
	}

	for (int d=0; d<DIM; d++) {
		m_loc[d] += m_v[d]*K; 
		if (m_loc[d] >= 2*M_PI) {
			m_loc[d] = m_loc[d] - 2*M_PI;  
		} else if (m_loc[d] <= 0) {
			m_loc[d] = 2*M_PI + m_loc[d]; 
		}
	}
}

double Particle::energy() {
	double e = 0; 
	for (int d=0; d<DIM; d++) {
		e += m_v[d]*m_v[d]; 
	}

	return .5*e; 
}
array<double,DIM> Particle::position() {return m_loc; }