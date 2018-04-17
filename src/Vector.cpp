#include "DataObjects.H"
#include "Timer.H"

#define ERROR(message) {\
	cout << "ERROR in " << __func__ << " (" << __FILE__ \
		<< " " << __LINE__ << ", r" << upcxx::rank_me() << "): " << message << endl; \
	upcxx::finalize(); \
	exit(0);}

Vector::Vector() {}

Vector::Vector(array<INT,DIM> dims, bool physical) {
	init(dims, physical); 
}

void Vector::init(array<INT,DIM> dims, bool physical) {
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
const Scalar& Vector::operator[](int c) const {return m_vector[c]; }

void Vector::forward() {
	for (int i=0; i<DIM; i++) {
		m_vector[i].forward(); 
	}
	setFourier(); 
}

void Vector::inverse() {
	for (int i=0; i<DIM; i++) {
		m_vector[i].inverse(); 
	}
	setPhysical(); 
}

void Vector::forward(Vector& a_vector) const {
	// deep copy 
	a_vector = (*this); 

	a_vector.forward(); 
}

void Vector::inverse(Vector& a_vector) const {
	// deep copy 
	a_vector = (*this); 

	a_vector.inverse(); 
}

Vector Vector::cross(Vector& a_v) const {
	Timer timer("cross product"); 
	if (isPhysical() || a_v.isPhysical()) {
		ERROR("must start in fourier space"); 
	} 

	// make copies and transform to physical space 
	Vector v1 = (*this); 
	Vector v2 = a_v; 
	v1.inverse(); 
	v2.inverse(); 

	Vector ret(m_dims, true);
	for (int i=0; i<v1[0].localSize(); i++) {
		ret[0][i] = v1[1][i]*v2[2][i] - v1[2][i]*v2[1][i]; 
		ret[1][i] = v1[2][i]*v2[0][i] - v1[0][i]*v2[2][i]; 
		ret[2][i] = v1[0][i]*v2[1][i] - v1[1][i]*v2[0][i]; 
	} 

	// transform to fourier space 
	ret.forward(); 

	return ret; 
}

Vector Vector::curl() const {
	if (!isFourier()) ERROR("must start in fourier space"); 
	Timer timer("curl"); 

	Vector x = (*this)[0].gradient(); 
	Vector y = (*this)[1].gradient(); 
	Vector z = (*this)[2].gradient(); 
	Vector curl(m_dims, false); 

	for (int i=0; i<m_vector[0].localSize(); i++) {
		curl[0][i] = z[1][i] - y[2][i]; 
		curl[1][i] = x[2][i] - z[0][i]; 
		curl[2][i] = y[0][i] - x[1][i]; 
	}

	return curl; 
}

Scalar Vector::divergence() const {
	if (!isFourier()) ERROR("must start in fourier space"); 
	Timer timer("divergence"); 

	// return scalar in fourier space 
	Scalar div(m_dims, false); 

	array<INT,DIM> i = {0,0,0}; 
	array<INT,DIM> start = m_vector[0].getPStart(); 
	array<INT,DIM> end = m_vector[0].getPEnd(); 
	array<double,DIM> k; 
	cdouble imag(0,1.); 
	for (i[2]=start[2]; i[2]<end[2]; i[2]++) {
		for (i[1]=start[1]; i[1]<end[1]; i[1]++) {
			for (i[0]=start[0]; i[0]<end[0]; i[0]++) {
				k = m_vector[0].freq(i); 
				div[i] = 0; 
				for (int d=0; d<DIM; d++) {
					div[i] += imag*k[d]*(*this)[d][i]; 
				}
			}
		}
	}
	return div; 
}

Vector Vector::laplacian() const {
	if (!isFourier()) ERROR("must start in fourier space"); 
	Timer timer("vector laplacian"); 

	Vector lap(m_dims, false); 
	array<INT,DIM> i = {0,0,0}; 
	array<INT,DIM> start = m_vector[0].getPStart(); 
	array<INT,DIM> end = m_vector[0].getPEnd(); 
	array<double,DIM> k; 
	for (i[2]=start[2]; i[2]<end[2]; i[2]++) {
		for (i[1]=start[1]; i[1]<end[1]; i[1]++) {
			for (i[0]=start[0]; i[0]<end[0]; i[0]++) {
				k = m_vector[0].freq(i); 
				for (int d=0; d<DIM; d++) {
					lap[d][i] = -(k[0]*k[0]+k[1]*k[1]+k[2]*k[2])*(*this)[d][i]; 
				}
			}
		}
	}
	return lap; 
}

bool Vector::isFourier() const {
	bool fourier = true; 
	for (int i=0; i<DIM; i++) {
		if (!m_vector[i].isFourier()) fourier = false; 
	}
	return fourier; 
}

bool Vector::isPhysical() const {
	bool physical = true; 
	for (int i=0; i<DIM; i++) {
		if (!m_vector[i].isPhysical()) physical = false; 
	}
	return false; 
}

// array<cdouble*,DIM> Vector::getLocal() {
// 	array<cdouble*,DIM> ret; 
// 	for (int i=0; i<DIM; i++) {
// 		ret[i] = m_vector[i].getLocal(); 
// 	}
// 	return ret; 
// }

array<INT,DIM> Vector::getDims() const {return m_vector[0].getDims(); }
array<INT,DIM> Vector::getPDims() const {return m_vector[0].getPDims(); }
array<INT,DIM> Vector::getPStart() const {return m_vector[0].getPStart(); }
array<INT,DIM> Vector::getPEnd() const {return m_vector[0].getPEnd(); }

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