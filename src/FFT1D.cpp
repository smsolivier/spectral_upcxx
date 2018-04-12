#include "FFT1D.H"
#include <iostream>
#ifdef FFTW 
#include "fftw3.h" 
#endif

FFT1D::FFT1D() {}

void FFT1D::forward(complex<double>* input, int N, int stride) {
	transform(input, N, stride, FFTW_FORWARD); 
}

void FFT1D::inverse(complex<double>* input, int N, int stride) {
	transform(input, N, stride, FFTW_BACKWARD); 

	// divide by N 
	for (int i=0; i<N; i++) {
		input[i*stride] /= N; 
	}
}

void FFT1D::transform(complex<double>* input, int N, int stride, int DIR) {
	fftw_plan plan; 
	fftw_complex* in; 
	in = reinterpret_cast<fftw_complex*>(input); 
	plan = fftw_plan_many_dft(
		1, // dimension of FFT 
		&N, // size of array 
		1, // number of FFTs 
		in, // input pointer 
		NULL, // inembed 
		stride, // input stride 
		0, // idist 
		in, // output pointer 
		NULL, // onembed 
		stride, // output stride 
		0, // odist 
		DIR, // transform sign 
		FFTW_ESTIMATE // FFTW flags 
		); 
	fftw_execute(plan); 
	fftw_destroy_plan(plan); 
}