#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include "DataObjects.H"

/// Marker particle class
class Particle {
public:
	/// default constructor 
	Particle(); 
	void move(const Vector& velocity, double K); 

	double energy(); 
	array<double,DIM> position(); 
private:
	array<double,DIM> m_loc; 
	array<double,DIM> m_v; 
}; 

#endif