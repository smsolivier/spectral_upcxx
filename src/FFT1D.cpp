#include "FFT1D.H"
#include <iostream>

FFT1D::FFT1D() {
	m_plan = NULL; 
}

FFT1D::FFT1D(int N, int stride, int DIR) {
	m_N = N; 
	m_stride = stride; 
	m_dir = DIR; 
	m_plan = NULL; 
}

FFT1D::~FFT1D() {
	fftw_destroy_plan(m_plan); 
}

void FFT1D::forward(cdouble* input, int N, int stride) {
	transform(input, N, stride, FFTW_FORWARD); 
}

void FFT1D::inverse(cdouble* input, int N, int stride) {
	transform(input, N, stride, FFTW_BACKWARD); 

	// divide by N 
	for (int i=0; i<N; i++) {
		input[i*stride] /= N; 
	}
}

void FFT1D::transform(cdouble* input, int N, int stride, int DIR) {
	fftw_complex* in; 
	in = reinterpret_cast<fftw_complex*>(input); 
	fftw_plan plan; 
	#pragma omp critical 
	{
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
	}
	fftw_execute(plan); 
	fftw_destroy_plan(plan); 
}

void FFT1D::transform(cdouble* input) {
	fftw_complex* in = reinterpret_cast<fftw_complex*>(input); 

	if (m_plan == NULL) {
		fftw_complex* test = new fftw_complex[m_N*m_stride]; 
		#pragma omp critical 
		m_plan = fftw_plan_many_dft(
			1, // dimension of FFT 
			&m_N, // size of array 
			1, // number of FFTs 
			test, // input pointer 
			NULL, // inembed 
			m_stride, // input stride 
			0, // idist 
			test, // output pointer 
			NULL, // onembed 
			m_stride, // output stride 
			0, // odist 
			m_dir, // transform sign 
			FFTW_MEASURE // FFTW flags 
			); 
		delete[] test; 
	} 
	fftw_execute_dft(m_plan, in, in); 
}