#include "DataObjects.H"
#include "CH_Timer.H"

#define ERROR(message) {\
	cout << "ERROR in " << __func__ << " (" << __FILE__ << " " << __LINE__ << "): " << message << endl; \
	upcxx::finalize(); \
	exit(0);}

Scalar operator*(const Scalar& a, const Scalar& b) {
	CH_TIMERS("scalar multiplication"); 
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Scalar ret(a.getDims(), a.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<a.localSize(); i++) {
		ret[i] = a[i]*b[i]; 
	}
	return ret; 
}

Scalar operator-(const Scalar& a, const Scalar& b) {
	CH_TIMERS("scalar subtraction"); 
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Scalar ret(a.getDims(), a.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<a.localSize(); i++) {
		ret[i] = a[i]-b[i]; 
	}
	return ret; 
}

Scalar operator+(const Scalar& a, const Scalar& b) {
	CH_TIMERS("scalar addition"); 
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Scalar ret(a.getDims(), a.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<a.localSize(); i++) {
		ret[i] = a[i]+b[i]; 
	}
	return ret; 
}

Vector operator-(const Vector& a, const Vector& b) {
	CH_TIMERS("vector subtraction"); 
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Vector ret(a.getDims(), a.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<DIM; i++) {
		for (int j=0; j<a[i].localSize(); j++) {
			ret[i][j] = a[i][j] - b[i][j]; 
		}
	}
	return ret; 
}

Vector operator+(const Vector& a, const Vector& b) {
	CH_TIMERS("vector addition"); 
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Vector ret(a.getDims(), a.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<DIM; i++) {
		for (int j=0; j<a[i].localSize(); j++) {
			ret[i][j] = a[i][j] + b[i][j]; 
		}
	}
	return ret; 
}

Scalar operator*(double alpha, const Scalar& s) {
	CH_TIMERS("double scalar multiplication"); 
	Scalar ret(s.getDims(), s.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<s.localSize(); i++) {
		ret[i] = alpha*s[i]; 
	}

	return ret; 
}

Vector operator*(double alpha, const Vector& v) {
	CH_TIMERS("double vector multiplication"); 
	Vector ret(v.getDims(), v.isPhysical()); 

	#pragma omp parallel for 
	for (int d=0; d<DIM; d++) {
		for (int i=0; i<v[d].localSize(); i++) {
			ret[d][i] = alpha*v[d][i]; 
		}
	}
	return ret; 
}