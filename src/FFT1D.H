#ifndef __FFT_1D_H__
#define __FFT_1D_H__

#include <vector> 
#include <complex> 
#include <array> 

using namespace std; 

typedef complex<double> cdouble; 

/// do FFT in 1 dimension 
class FFT1D {
public:
	/// default constructor 
	FFT1D(); 
	/// in place forward transform on data of size N with stride stride 
	void forward(cdouble* input, int N, int stride=1); 
	/// in place inverse transform on data of size N with stride stride 
	void inverse(cdouble* input, int N, int stride=1); 
	/// in place transform 
	/** dir = FFTW_FORWARD is forward \n
		dir = FFTW_BACKWARD is inverse 
	**/ 
	void transform(cdouble* input, int N, int stride, int dir);
private: 
}; 

#endif