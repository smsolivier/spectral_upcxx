#include "FFT1D.H"
#include <iostream>

FFT1D::FFT1D() {
	m_forward = NULL; 
	m_backward = NULL; 
}

void FFT1D::init(int N, int stride, cdouble* input) {
	m_N = N; 
	m_stride = stride; 
	fftw_complex* in = reinterpret_cast<fftw_complex*>(input); 
#ifdef MEASURE
	int measure = FFTW_MEASURE; 
#else
	int measure = FFTW_ESTIMATE; 
#endif
	m_forward = fftw_plan_many_dft(
		1, // dimension of FFT 
		&m_N, // size of array 
		1, // number of FFTs 
		in, // input pointer 
		NULL, // inembed 
		m_stride, // input stride 
		0, // idist 
		in, // output pointer 
		NULL, // onembed 
		m_stride, // output stride 
		0, // odist 
		1, // transform sign 
		measure // FFTW flags 
		); 
	m_backward = fftw_plan_many_dft(
		1, // dimension of FFT 
		&m_N, // size of array 
		1, // number of FFTs 
		in, // input pointer 
		NULL, // inembed 
		m_stride, // input stride 
		0, // idist 
		in, // output pointer 
		NULL, // onembed 
		m_stride, // output stride 
		0, // odist 
		-1, // transform sign 
		measure // FFTW flags 
		);
	m_alignment = fftw_alignment_of((double*)in); 
}

FFT1D::~FFT1D() {
	if (m_forward != NULL) fftw_destroy_plan(m_forward); 
	if (m_backward != NULL) fftw_destroy_plan(m_backward); 
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

void FFT1D::transform(cdouble* input, int DIR) {
	fftw_complex* in = reinterpret_cast<fftw_complex*>(input);
#ifdef DEBUG 
	if (fftw_alignment_of((double*)in) != m_alignment) cout << "unaligned" << endl; 
#endif
	if (DIR==1) fftw_execute_dft(m_forward, in, in); 
	else fftw_execute_dft(m_backward, in, in); 
}