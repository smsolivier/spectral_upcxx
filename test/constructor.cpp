#include "DataObjects.H"
#include "Timer.H"

using namespace std; 

int main() {
	upcxx::init(); 
	START_WTIMER(); 

	INT N = 16; 
	array<INT,DIM> dims = {N, N, N}; 
	Vector x(dims); 
	Vector y = x; 
	upcxx::finalize(); 
}