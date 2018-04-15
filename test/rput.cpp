#include <upcxx/upcxx.hpp> 
#include <vector> 

using namespace std; 

int main () {
	upcxx::init(); 
	int N = 4; 
	vector<upcxx::global_ptr<int>> ptr(upcxx::rank_n()); 
	ptr[upcxx::rank_me()] = upcxx::new_array<int>(N); 
	for (int i=0; i<upcxx::rank_n(); i++) {
		ptr[i] = upcxx::broadcast(ptr[i], i).wait(); 
	}

	int* vals = new int[N]; 
	for (int i=0; i<N; i++) {
		vals[i] = i; 
	}

	if (upcxx::rank_me() == 0) {
		#pragma omp parallel 
		{
			upcxx::future<> f = upcxx::make_future(); 
			#pragma omp for 
			for (int i=0; i<N; i++) {
				f = upcxx::when_all(f, upcxx::rput(vals+i, ptr[1]+i, 1)); 
			}
			f.wait(); 			
		}
	}

	upcxx::barrier(); 
	if (upcxx::rank_me() == 1) {
		int* local = ptr[1].local(); 
		for (int i=0; i<N; i++) {
			cout << local[i] << endl; 
		}
	}
	upcxx::finalize(); 
}