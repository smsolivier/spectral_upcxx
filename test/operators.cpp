#include "DataObjects.H"
#include "Writer.H"
#include <iostream> 
#include <omp.h>

using namespace std; 

int main(int argc, char* argv[]) {
	upcxx::init(); 
	INT N = 32; 
	if (argc > 1) N = atoi(argv[1]); 
	array<INT,DIM> dims = {N, N, N}; 
	Writer writer("solution"); 
	// gradient 
	Scalar s(dims); 
	writer.add(s, "sin"); 
	array<INT,DIM> start = s.getPStart(); 
	array<INT,DIM> end = s.getPEnd(); 
	array<INT,DIM> ind = {0,0,0}; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				s[ind] = sin(2*M_PI/N*ind[0]); 
				// s[ind] = sin(2*M_PI/N*ind[0])*sin(2*M_PI/N*ind[1])*sin(2*M_PI/N*ind[2]); 
			}
		}
	}
	Scalar s_f(dims); 
	s.forward(s_f); 

	// check addition
	Scalar sum = s + s; 
	bool wrong= false; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				if (abs(sum[ind].real() - 2*sin(2*M_PI/N*ind[0])) > 1e-3) {
					wrong = true;
				}
			}
		}
	}
	if (wrong) cout << "addition failed" << endl; 
	else cout << "addition passed" << endl; 

	Scalar sub = s - s; 
	wrong= false; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				if (abs(sub[ind].real()) > 1e-3) {
					wrong = true;
				}
			}
		}
	}
	if (wrong) cout << "subtraction failed" << endl; 
	else cout << "subtraction passed" << endl; 

	// check gradient 
	Vector grad = s_f.gradient(); 
	grad.inverse(); 
	writer.add(grad, "grad"); 
	wrong = false; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				if (abs(grad[0][ind].real() - cos(2*M_PI/N*ind[0])) > 1e-3) {
					wrong = true;
				} else if (abs(grad[1][ind].real()) > 1e-3) {
					wrong = true; 
				} else if (abs(grad[2][ind].real()) > 1e-3) {
					wrong = true; 
				}
			}
		}
	}
	if (wrong) cout << "gradient failed" << endl; 
	else cout << "gradient passed" << endl; 

	// check laplacian 
	Scalar lap = s_f.laplacian(); 
	lap.inverse(); 
	wrong = false; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				if (abs(lap[ind].real() + sin(2*M_PI/N*(double)ind[0])) > 1e-3) {
					wrong = true;
				}
			}
		}
	}
	if (wrong) cout << "laplacian failed" << endl; 
	else cout << "laplacian passed" << endl; 

	Scalar lap_inv; 
	lap.forward(lap_inv); 
	lap_inv.laplacian_inverse(); 
	lap_inv.inverse(); 
	wrong = false; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				if (abs(lap_inv[ind].real() - s[ind].real()) > 1e-3) {
					wrong = true;
				}
			}
		}
	}
	if (wrong) cout << "laplacian inverse failed" << endl; 
	else cout << "laplacian inverse passed" << endl; 

	// test divergence 
	Vector v(dims); 
	v[0] = s; 
	v.forward(); 
	Scalar div = v.divergence(); 
	div.inverse(); 
	wrong = false; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				if (abs(div[ind].real() - cos(2*M_PI/N*(double)ind[0])) > 1e-3) {
					wrong = true;
				}
			}
		}
	}
	if (wrong) cout << "divergence failed" << endl; 
	else cout << "divergence passed" << endl; 

	Vector vlap = v.laplacian(); 
	vlap.inverse(); 
	wrong = false; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				if (abs(vlap[0][ind].real() + sin(2*M_PI/N*(double)ind[0])) > 1e-3) {
					wrong = true;
				}
			}
		}
	}
	if (wrong) cout << "vector laplacian failed" << endl; 
	else cout << "vector laplacian passed" << endl; 

	Vector v1(dims); 
	Vector v2(dims); 
	for (int i=0; i<v1[0].localSize(); i++) {
		v1[2][i] = 1.; 
		v2[1][i] = 1.; 
	}
	v1.forward(); 
	v2.forward(); 
	Vector cross = v1.cross(v2); 
	cross.inverse(); 
	wrong = false; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				if (abs(cross[0][ind].real() + 1) > 1e-3) {
					wrong = true;
				} else if (abs(cross[1][ind].real()) > 1e-3) wrong = true; 
				else if (abs(cross[2][ind].real() > 1e-3)) wrong = true; 
			}
		}
	}
	if (wrong) cout << "cross failed" << endl; 
	else cout << "cross passed" << endl;

	grad.forward(); 
	Vector curl = grad.curl(); 
	curl.inverse(); 
	writer.add(curl, "curl"); 
	wrong = false; 
	for (int d=0; d<DIM; d++) {
		for (int i=0; i<curl[0].localSize(); i++) {
			if (abs(curl[d][i].real()) > 1e-3) wrong = true; 
		}
	}
	if (wrong) cout << "curl failed" << endl; 
	else cout << "curl passed" << endl; 

	curl.forward(); 
	Vector mult = 2.*curl; 
	wrong = false; 
	for (ind[2]=start[2]; ind[2]<end[2]; ind[2]++) {
		for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
			for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
				for (int d=0; d<DIM; d++) {
					if (abs(mult[d][ind] - 2.*curl[d][ind]) > 1e-3) wrong = true; 
				} 
			}
		}
	}
	if (wrong) cout << "double * vector failed" << endl; 
	else cout << "double * vector passed" << endl;

	upcxx::finalize(); 
}