#ifndef __FFT_1D_H__
#define __FFT_1D_H__

#include <vector> 
#include <complex> 
#include <array> 
#ifdef FFTW 
#include "fftw3.h" 
#endif

using namespace std; 

typedef complex<double> cdouble; 
typedef uint64_t INT; 

/// do FFT in 1 dimension 
class FFT1D {
public:
	/// default constructor 
	FFT1D(); 
	/// set stride and size to reuse plan 
	FFT1D(int N, int stride, int DIR); 
	/// destructor 
	~FFT1D(); 
	/// in place forward transform on data of size N with stride stride 
	void forward(cdouble* input, int N, int stride=1); 
	/// in place inverse transform on data of size N with stride stride 
	void inverse(cdouble* input, int N, int stride=1); 
	/// in place transform 
	/** dir = FFTW_FORWARD is forward \n
		dir = FFTW_BACKWARD is inverse 
	**/ 
	void transform(cdouble* input, int N, int stride, int dir);
	void transform(cdouble* input); 
private: 
	/// fftw plan for many FFTs of the same size and stride 
	fftw_plan m_plan; 
	/// size of FFT 
	int m_N; 
	/// stride 
	int m_stride; 
	/// FFT direction 
	int m_dir; 
}; 

#endif