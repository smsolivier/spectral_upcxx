#include "DataObjects.H"

#define ERROR(message) {\
	cout << "ERROR in " << __func__ << " (" << __FILE__ << " " << __LINE__ << "): " << message << endl; \
	upcxx::finalize(); \
	exit(0);}

Scalar operator*(const Scalar& a, const Scalar& b) {
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Scalar ret(a.getDims(), a.isPhysical()); 

	for (int i=0; i<a.localSize(); i++) {
		ret[i] = a[i]*b[i]; 
	}
	return ret; 
}

Scalar operator-(const Scalar& a, const Scalar& b) {
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Scalar ret(a.getDims(), a.isPhysical()); 

	for (int i=0; i<a.localSize(); i++) {
		ret[i] = a[i]+b[i]; 
	}
	return ret; 
}

Scalar operator+(const Scalar& a, const Scalar& b) {
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Scalar ret(a.getDims(), a.isPhysical()); 

	for (int i=0; i<a.localSize(); i++) {
		ret[i] = a[i]+b[i]; 
	}
	return ret; 
}

Vector operator-(const Vector& a, const Vector& b) {
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Vector ret(a.getDims(), a.isPhysical()); 

	for (int i=0; i<DIM; i++) {
		for (int j=0; j<a[i].localSize(); j++) {
			ret[i][j] = a[i][j] - b[i][j]; 
		}
	}
	return ret; 
}

Vector operator+(const Vector& a, const Vector& b) {
	if (a.isPhysical() != b.isPhysical()) ERROR("space mismatch"); 
	Vector ret(a.getDims(), a.isPhysical()); 

	for (int i=0; i<DIM; i++) {
		for (int j=0; j<a[i].localSize(); j++) {
			ret[i][j] = a[i][j] + b[i][j]; 
		}
	}
	return ret; 
}

Scalar operator*(double alpha, const Scalar& s) {
	Scalar ret(s.getDims(), s.isPhysical()); 

	for (int i=0; i<s.localSize(); i++) {
		ret[i] = alpha*s[i]; 
	}

	return ret; 
}

Vector operator*(double alpha, const Vector& v) {
	Vector ret(v.getDims(), v.isPhysical()); 

	for (int d=0; d<DIM; d++) {
		for (int i=0; i<v[i].localSize(); i++) {
			ret[d][i] = alpha*v[d][i]; 
		}
	}
	return ret; 
}