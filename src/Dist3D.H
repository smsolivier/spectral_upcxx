#ifndef __DIST_3D_H__ 
#define __DIST_3D_H__

#include "FFT1D.H"
#include <array> 
#include <vector> 
#include <upcxx/upcxx.hpp> 
#include <complex> 

using namespace std; 

// typedef std::complex<double> cdouble; 
typedef complex<double> cdouble; 

/// Store a 3D array with UPCXX 
/** stores x,y locally and distributes in z **/ 
class Dist3D {
public: 
	/// default constructor 
	Dist3D(); 
	/// constructor 
	Dist3D(array<int,DIM> N); 
	/// initialize the array after calling default constructor 
	void init(array<int,DIM> N); 
	/// set a value in the distributed array 
	void set(array<int,DIM> index, cdouble val); 
	/// get an element from the distributed array 
	cdouble operator[](array<int,DIM> index); 

	/// in place forward transform 
	void forward(); 
	/// in place reverse transform 
	void inverse(); 

	/// get size per processor 
	int sizePerProcessor(); 
	/// get total size 
	int size(); 
	/// get pointer to local data
	cdouble* getLocal(); 
private:
	/// fourier transform 
	void transform(int dir); 
	/// global transpose 
	void transpose(); 
	/// get the rank and location into the distributed array 
	void getIndex(array<int,DIM> index, int& rank, int& loc); 
	/// number of Nx by Ny slabs in the z direction 
	int m_Nz;
	/// total number of values \f$N_x \times N_y \times N_z \f$  
	int m_N; 
	/// size of data owned by a processor 
	int m_dSize; 
	/// dimensions of array in x, y, z 
	array<int,DIM> m_dims; 
	/// global pointers to z slabs 
	vector<upcxx::global_ptr<cdouble>> m_ptrs; 
	/// local version of m_ptrs[rank_me()] 
	cdouble* m_local; 
	/// object for 1D ffts 
	FFT1D m_fft; 
}; 

#endif