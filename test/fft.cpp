#include "FFT1D.H"
#include <iostream> 
#include <complex> 
#include <fstream> 
#include <chrono> 

using namespace std; 

void transform(cdouble* f, int N, int DIR) {
	#pragma omp parallel 
	{
		FFT1D fft(N, 1, DIR); 
		#pragma omp for 
		for (int k=0; k<N; k++) {
			for (int j=0; j<N; j++) {
				cdouble* start = f + j*N + k*N*N; 
				fft.transform(start); 
				// fft.transform(start, N, 1, DIR); 
			}
		}
	}

	#pragma omp parallel 
	{
		FFT1D fft(N, N, DIR); 
		#pragma omp for 
		for (int k=0; k<N; k++) {
			for (int i=0; i<N; i++) {
				cdouble* start = f + i + k*N*N; 
				fft.transform(start); 
				// fft.transform(start, N, N, DIR); 
			}
		}
	}

	#pragma omp parallel 
	{
		FFT1D fft(N, N*N, DIR); 
		#pragma omp for 
		for (int j=0; j<N; j++) {
			for (int i=0; i<N; i++) {
				cdouble* start = f + i + j*N; 
				fft.transform(start); 
				// fft.transform(start, N, N*N, DIR); 
			}
		}
	}
}

int main(int argc, char* argv[]) {
	int N = 32; 
	if (argc > 1) N = atoi(argv[1]); 
	int N3 = pow(N, 3); 

	cdouble* f = new cdouble[N3]; 
	cdouble* g = new cdouble[N3]; 

	for (int i=0; i<N3; i++) {
		f[i] = 1.; 
		g[i] = f[i]; 
	}

	// cdouble* row = new cdouble[N]; 
	// for (int i=0; i<N; i++) {
	// 	row[i] = 1.; 
	// }
	// FFT1D forward(N, 1, 1); 
	// forward.transform(row); 
	// FFT1D backward(N, 1, -1); 
	// backward.transform(row); 
	// for (int i=0; i<N; i++) {
	// 	row[i] /= N; 
	// 	if (abs(row[i].real() - 1.) > 1e-3) cout << row[i] << endl; 
	// }

	chrono::time_point<chrono::system_clock> start = chrono::system_clock::now(); 
	transform(f, N, 1); 
	transform(f, N, -1); 
	chrono::duration<double> el = chrono::system_clock::now() - start; 

	cout << "time = " << el.count() << " seconds" << endl; 

	#pragma omp parallel for 
	for (int i=0; i<N3; i++) {
		f[i] /= N3; 
	}

	bool wrong = false; 
	for (int i=0; i<N3; i++) {
		if (abs(f[i].real() - g[i].real()) > 1e-3) wrong = true; 
	}
	if (wrong) cout << "WRONG!" << endl; 

	delete[] f; 
	delete[] g; 
}