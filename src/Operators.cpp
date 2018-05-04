#include "DataObjects.H"
#include "CH_Timer.H"

#define ERROR(message) {\
	cout << "ERROR in " << __func__ << " (" << __FILE__ << " " << __LINE__ << "): " << message << endl; \
	upcxx::finalize(); \
	exit(0);}

Scalar operator*(const Scalar& a, const Scalar& b) {
	CH_TIMERS("Scalar *"); 
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Scalar ret(a.getDims(), a.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<a.localSize(); i++) {
		ret[i] = a[i]*b[i]; 
	}
	return ret; 
}

Scalar operator-(const Scalar& a, const Scalar& b) {
	CH_TIMERS("Scalar -"); 
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Scalar ret(a.getDims(), a.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<a.localSize(); i++) {
		ret[i] = a[i]-b[i]; 
	}
	return ret; 
}

Scalar operator+(const Scalar& a, const Scalar& b) {
	CH_TIMERS("Scalar +"); 
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Scalar ret(a.getDims(), a.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<a.localSize(); i++) {
		ret[i] = a[i]+b[i]; 
	}
	return ret; 
}

Vector operator-(const Vector& a, const Vector& b) {
	CH_TIMERS("Vector -"); 
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Vector ret(a.getDims(), a.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<a[0].localSize(); i++) {
		for (int d=0; d<DIM; d++) {
			ret[d][i] = a[d][i] - b[d][i]; 
		}
	}

	return ret; 
}

Vector operator+(const Vector& a, const Vector& b) {
	CH_TIMERS("Vector +"); 
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Vector ret(a.getDims(), a.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<a[0].localSize(); i++) {
		for (int d=0; d<DIM; d++) {
			ret[d][i] = a[d][i] + b[d][i]; 
		}
	}

	return ret; 
}

Scalar operator*(double alpha, const Scalar& s) {
	CH_TIMERS("double * Scalar"); 
	Scalar ret(s.getDims(), s.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<s.localSize(); i++) {
		ret[i] = alpha*s[i]; 
	}

	return ret; 
}

Vector operator*(double alpha, const Vector& v) {
	CH_TIMERS("double * Vector"); 
	Vector ret(v.getDims(), v.isPhysical()); 

	#pragma omp parallel for 
	for (int i=0; i<v[0].localSize(); i++) {
		for (int d=0; d<DIM; d++) {
			ret[d][i] = alpha*v[d][i]; 
		}
	}

	return ret; 
}