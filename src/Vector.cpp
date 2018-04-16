#include "DataObjects.H"

Vector::Vector() {} 

Vector::Vector(array<INT,DIM> dims, bool physical) {
	m_dims = dims; 

	// allocate scalars 
	for (int i=0; i<DIM; i++) {
		m_vector[i].init(m_dims, physical); 
	}

	// flag as physical/fourier 
	if (physical == true) setPhysical(); 
	else setFourier(); 

	// compute total size 
	m_N = 1; 
	for (int i=0; i<DIM; i++) {
		m_N *= m_dims[i]; 
	}
}

Scalar& Vector::operator[](int c) {return m_vector[c]; }

void Vector::forward() {
	for (int i=0; i<DIM; i++) {
		m_vector[i].forward(); 
	}
}

void Vector::inverse() {
	for (int i=0; i<DIM; i++) {
		m_vector[i].inverse(); 
	}
}

void Vector::forward(Vector& a_vector) {

}

void Vector::inverse(Vector& a_vector) {

}

bool Vector::isFourier() {
	bool fourier = true; 
	for (int i=0; i<DIM; i++) {
		if (!m_vector[i].isFourier()) fourier = false; 
	}
	return fourier; 
}

bool Vector::isPhysical() {
	bool physical = true; 
	for (int i=0; i<DIM; i++) {
		if (!m_vector[i].isPhysical()) physical = false; 
	}
	return false; 
}

array<cdouble*,DIM> Vector::getLocal() {
	array<cdouble*,DIM> ret; 
	for (int i=0; i<DIM; i++) {
		ret[i] = m_vector[i].getLocal(); 
	}
	return ret; 
}

array<INT,DIM> Vector::getDims() {return m_vector[0].getDims(); }
array<INT,DIM> Vector::getPDims() {return m_vector[0].getPDims(); }

void Vector::setPhysical() {
	for (int i=0; i<DIM; i++) {
		m_vector[i].setPhysical(); 
	}
}

void Vector::setFourier() {
	for (int i=0; i<DIM; i++) {
		m_vector[i].setFourier(); 
	}
}