#include <upcxx/upcxx.hpp> 
#include <vector> 
#include <omp.h>

using namespace std; 

/* 	creates and broadcasts a global array on each upcxx rank 
	attemps to send the local data from rank 0 to rank 1 inside 
		an OpenMP parallel region 
	checks that rank 1 now has the data from rank 0 

	Doesn't seg fault when: 
		OMP_NUM_THREADS=1 
		rput's are only called from the master OMP thread (uncomment if (tid == 0))

	Seg faults when:
		OMP_NUM_THREADS > 1 
		send from only the non master thread (uncomment if (tid == 1)) 
		rput's are wrapped in an #pragma omp critical region 

	the cout's suggest that it fails as soon as the non-master thread calls rput 
*/ 

int main () {
	upcxx::init(); 
	int N = 20; // size of array on each rank 

	// --- create global array and broadcast the pointer to all ranks --- 
	vector<upcxx::global_ptr<int>> ptrs(upcxx::rank_n()); 
	ptrs[upcxx::rank_me()] = upcxx::new_array<int>(N); 
	for (int i=0; i<upcxx::rank_n(); i++) {
		ptrs[i] = upcxx::broadcast(ptrs[i], i).wait(); 
	}
	upcxx::barrier(); 

	// initialize local to my rank 
	int* local = ptrs[upcxx::rank_me()].local(); 
	for (int i=0; i<N; i++) {
		local[i] = upcxx::rank_me(); 
	}

	// --- send rank 0's data to rank 1 in parallel with OpenMP --- 
	// if (upcxx::rank_me() == 0) {
		int orank = (upcxx::rank_me() + 1) % upcxx::rank_n(); 
		#pragma omp parallel 
		{
			int tid = omp_get_thread_num(); // OMP thread id 
			upcxx::future<> f = upcxx::make_future(); // setup future chain 

			// check futures can be made in parallel 
			#pragma omp critical
			cout << "made future on " << tid << endl; 

			// send messages from rank 0 to rank 1 in OMP for loop 
			#pragma omp for 
			for (int i=0; i<N; i++) {
				// output starting an rput 
				#pragma omp critical 
				cout << "starting rput on " << tid << endl; 

				// if (tid == 0) // no issues when only sent from thread 0 
				// if (tid == 1) // issues if only sending from thread 1 
				f = upcxx::when_all(f, upcxx::rput(local+i, ptrs[orank]+i, 1)); 

				#pragma omp critical 
				cout << "sent message " << i << " from " << tid << endl; 
			}

			// wait for the chained futures 
			f.wait(); 

			#pragma omp critical 
			cout << "waited on " << tid << endl; 

			// explicit OMP barrier just in case 
			#pragma omp barrier 

			#pragma omp critical 
			cout << tid << " is past the barrier" << endl; 
		} // end OMP parallel region 
	// }
	upcxx::barrier(); // make sure rank 0 done before checking for correctness 

	// --- ensure correctness --- 
	if (upcxx::rank_me() == 1) {
		bool wrong = false; 
		for (int i=0; i<N; i++) {
			if (local[i] != 0) wrong = true; 
		}

		if (wrong) cout << "WRONG!" << endl; 
		else cout << "correct" << endl; 
	}
	upcxx::finalize(); 
}