#include "fftw3.h" 
#include <complex> 
#include <chrono> 
#include <iostream>

using namespace std; 

typedef complex<double> cdouble; 

int main(int argc, char* argv[]) {
	int N =32; 
	if (argc > 1) N = atoi(argv[1]); 
	int nruns = 15; 

	cdouble* f = new cdouble[N*N*N]; 
	cdouble* ans = new cdouble[N*N*N]; 

	fftw_complex* fftw_f = reinterpret_cast<fftw_complex*>(f); 

	fftw_plan forward = fftw_plan_dft_3d(N, N, N, fftw_f, fftw_f, 
		FFTW_FORWARD, FFTW_MEASURE); 
	fftw_plan backward = fftw_plan_dft_3d(N, N, N, fftw_f, fftw_f, 
		FFTW_BACKWARD, FFTW_MEASURE); 

	for (int i=0; i<N*N*N; i++) {
		f[i] = (double)rand()/RAND_MAX; 
		ans[i] = f[i]; 
	}

	chrono::time_point<chrono::system_clock> start; 
	chrono::duration<double> el;
	double min = numeric_limits<double>::max(); 
	for (int i=0; i<nruns; i++) {
		start = chrono::system_clock::now(); 
		fftw_execute(forward); 
		fftw_execute(backward); 
		for (int j=0; j<N*N*N; j++) {
			f[j] /= N*N*N; 
		}
		el = chrono::system_clock::now() - start; 
		if (el.count() < min) min = el.count(); 
	}

	cout << "min time = " << min << " seconds" << endl;

	bool wrong = false; 
	for (int i=0; i<N*N*N; i++) {
		if (abs(f[i].real() - ans[i].real()) > 1e-3) {
			cout << f[i].real() << endl; 
			wrong = true; 
		}
	}
	if (wrong) cout << "WRONG!" << endl; 
	else cout << "my man!" << endl; 
}