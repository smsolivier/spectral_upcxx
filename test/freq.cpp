#include "DataObjects.H"
#include <iostream> 

using namespace std; 

int main() {
	upcxx::init(); 
	int N = 16; 
	Scalar s({N, N, N}, true); 

	array<double,DIM> k; 

	for (int i=0; i<N; i++) {
		k = s.freq({i,0,0}); 
		cout << k[0] << endl; 
	}
	upcxx::finalize(); 
}