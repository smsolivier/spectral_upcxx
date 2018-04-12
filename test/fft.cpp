#include "FFT1D.H"
#include <iostream> 
#include <complex> 
#include <fstream> 

using namespace std; 

typedef complex<double> cdouble; 

void write(cdouble* f, int N) {
	ofstream out("out"); 
	for (int i=0; i<N; i++) {
		for (int j=0; j<N; j++) {
			out << f[j+i*N].real() << " "; 
		}
		out << endl; 
	}
	out.close(); 
}

int main() {
	int N = 32; 
	FFT1D fft; 

	cdouble* f = new cdouble[N*N]; 
	double h = 2*M_PI/N;
	for (int i=0; i<N; i++) {
		for (int j=0; j<N; j++) {
			f[j+i*N] = sin(i*h)*sin(j*h); 
			// f[j+i*N] = sin(i*h); 
			// f[j+i*N] = 1; 
		}
	}

	for (int i=0; i<N; i++) {
		fft.forward(f+i*N, N, 1); 
	}

	for (int i=0; i<N; i++) {
		fft.forward(f+i, N, N); 
	}

	for (int i=0; i<N; i++) {
		fft.inverse(f+i*N, N, 1); 
	}

	for (int i=0; i<N; i++) {
		fft.inverse(f+i, N, N); 
	}

	write(f, N); 

	delete(f); 
}