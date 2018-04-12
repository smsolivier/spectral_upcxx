#include <upcxx/upcxx.hpp>
#include <chrono>
#ifdef OMP 
#include <omp.h>
#endif

using namespace std; 

int hit() {
	double x = (double)rand()/RAND_MAX; 
	double y = (double)rand()/RAND_MAX; 
	if (x*x + y*y <= 1.) return 1; 
	else return 0; 
}

int main(int argc, char* argv[]) {
	upcxx::init(); 
	system("hostname"); 

	double s = 0; 
	int N = 100000; 
	if (argc > 1) N = atoi(argv[1]); 
	srand(upcxx::rank_me()); 
	chrono::time_point<chrono::system_clock> start = chrono::system_clock::now(); 
#ifdef OMP
	int nthreads;
	#pragma omp parallel 
	{
		#pragma omp master
		nthreads = omp_get_num_threads(); 
		double count = 0; 
		#pragma omp for schedule(static)
		for (int i=0; i<N/upcxx::rank_n(); i++) {
			count += hit(); 
		}

		#pragma omp critical 
		{
			s += count; 
		}
	}
	cout << "nthreads = " << nthreads << endl; 
#else
	for (int i=0; i<N; i++) {
		s += hit(); 
	}
#endif
	chrono::duration<double> el = chrono::system_clock::now() - start; 

	int allhits = upcxx::allreduce(s, plus<int>()).wait(); 

	if (upcxx::rank_me() == 0) {
		cout << 4.*allhits/N << endl; 
		cout << "Loop Time = " << el.count() << endl; 
	}

	upcxx::finalize(); 
}