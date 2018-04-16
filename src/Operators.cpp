#include "DataObjects.H"

Scalar operator*(double alpha, Scalar s) {
	Scalar ret(s.getDims(), s.isPhysical()); 

	for (int i=0; i<s.localSize(); i++) {
		ret[i] = alpha*s[i]; 
	}

	return ret; 
}