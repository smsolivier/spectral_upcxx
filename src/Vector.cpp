#include "DataObjects.H"
#include "CH_Timer.H"

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

Vector Vector::cross(const Vector& a_v) const {
	CH_TIMERS("cross product"); 
	if (isPhysical() || a_v.isPhysical()) {
		ERROR("must start in fourier space"); 
	} 

	// make copies and transform to physical space 
	Vector u; 
	Vector v; 
	inverse(u); 
	a_v.inverse(v);  

	Vector ret(m_dims, true);
	#pragma omp parallel for 
	for (int i=0; i<u[0].localSize(); i++) {
		ret[0][i] = u[1][i]*v[2][i] - u[2][i]*v[1][i]; 
		ret[1][i] = u[2][i]*v[0][i] - u[0][i]*v[2][i]; 
		ret[2][i] = u[0][i]*v[1][i] - u[1][i]*v[0][i]; 
	} 

	// transform to fourier space 
	ret.forward(); 

	return ret; 
}

Vector Vector::curl() const {
	CH_TIMERS("curl"); 
	if (!isFourier()) ERROR("must start in fourier space"); 

	Vector curl(m_dims, false);  
	#pragma omp parallel 
	{
		array<INT,DIM> start = getPStart(); 
		array<INT,DIM> end = getPEnd(); 
		array<INT,DIM> ind = {0,0,0}; 
		array<double,DIM> f; 
		cdouble imag(0,1.); 
		for (INT k=start[2]; k<end[2]; k++) {
			for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
				for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
					ind[2] = k; 
					f = m_vector[0].freq(ind); 
					curl[0][ind] = imag*f[1]*(*this)[2][ind] 
						- imag*f[2]*(*this)[1][ind]; 
					curl[1][ind] = imag*f[2]*(*this)[0][ind] - 
						imag*f[0]*(*this)[2][ind]; 
					curl[2][ind] = imag*f[0]*(*this)[1][ind] - 
						imag*f[1]*(*this)[0][ind]; 
				}
			}
		}
	}
	
	return curl; 
}

Scalar Vector::divergence() const {
	CH_TIMERS("divergence"); 
	if (!isFourier()) ERROR("must start in fourier space"); 

	// return scalar in fourier space 
	Scalar div(m_dims, false); 

	#pragma omp parallel 
	{
		array<INT,DIM> i = {0,0,0}; 
		const array<INT,DIM> start = m_vector[0].getPStart(); 
		const array<INT,DIM> end = m_vector[0].getPEnd(); 
		array<double,DIM> f; 
		cdouble imag(0,1.); 
		#pragma omp for
		for (INT k=start[2]; k<end[2]; k++) {
			for (i[1]=start[1]; i[1]<end[1]; i[1]++) {
				for (i[0]=start[0]; i[0]<end[0]; i[0]++) {
					i[2] = k; 
					f = m_vector[0].freq(i); 
					div[i] = 0; 
					for (int d=0; d<DIM; d++) {
						div[i] += imag*f[d]*(*this)[d][i]; 
					}
				}
			}
		}
	}
	return div; 
}

Vector Vector::laplacian() const {
	CH_TIMERS("vector laplacian"); 
	if (!isFourier()) ERROR("must start in fourier space"); 

	Vector lap(m_dims, false); 
	
	#pragma omp parallel
	{
		array<INT,DIM> i = {0,0,0}; 
		array<INT,DIM> start = m_vector[0].getPStart(); 
		array<INT,DIM> end = m_vector[0].getPEnd(); 
		array<double,DIM> f; 
		#pragma omp parallel for 
		for (INT k=start[2]; k<end[2]; k++) {
			for (i[1]=start[1]; i[1]<end[1]; i[1]++) {
				for (i[0]=start[0]; i[0]<end[0]; i[0]++) {
					i[2] = k; 
					f = m_vector[0].freq(i); 
					for (int d=0; d<DIM; d++) {
						lap[d][i] = -(f[0]*f[0]+f[1]*f[1]+f[2]*f[2])*(*this)[d][i]; 
					}
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

INT Vector::localSize() const {return m_vector[0].localSize(); }

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