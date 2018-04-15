#ifndef __SCALAR_H__ 
#define __SCALAR_H__

#include "FFT1D.H"
#include <array> 
#include <vector> 
#include <upcxx/upcxx.hpp> 
#include <complex> 

using namespace std; 

/// Store a 3D array with UPCXX 
/** stores x,y locally and distributes in z **/ 
class Scalar {
public: 
	/// default constructor 
	Scalar(); 
	/// constructor 
	Scalar(array<INT,DIM> N); 
	/// destructor 
	~Scalar(); 
	/// initialize the array after calling default constructor 
	void init(array<INT,DIM> N); 
	/// set a value in the distributed array 
	void set(array<INT,DIM> index, cdouble val); 
	/// get an element from the distributed array 
	cdouble operator[](array<INT,DIM> index); 
	/// direct access to local pointer 
	cdouble& operator[](INT index); 

	/// in place forward transform 
	void forward(); 
	/// in place reverse transform 
	void inverse(); 

	/// element wise addition with another Scalar 
	void add(Scalar& a); 

	/// get size per processor 
	INT sizePerProcessor(); 
	/// get total size 
	INT size(); 
	/// get pointer to local data
	cdouble* getLocal(); 
	/// compute allocated memory size per processor 
	double memory(); 
private:
	/// fourier transform 
	void transform(int dir); 
	/// transpose so that contiguous dimension is in z  
	void transposeX2Z(cdouble* f); 
	/// transpose from contiguous in z to contiguous in x 
	void transposeZ2X(cdouble* f); 
	/// get the rank and location into the distributed array 
	void getIndex(array<INT,DIM> index, INT& rank, INT& loc); 
	/// number of Nx by Ny slabs in the z direction 
	INT m_Nz;
	/// number of Nx by Nz slabs in the y direction (for after the global transpose) 
	INT m_Ny; 
	/// total number of values \f$N_x \times N_y \times N_z \f$  
	INT m_N; 
	/// size of data owned by a processor 
	INT m_dSize; 
	/// dimensions of array in x, y, z 
	array<INT,DIM> m_dims; 
	/// global pointers to z slabs 
	vector<upcxx::global_ptr<cdouble>> m_ptrs; 
	/// local version of m_ptrs[rank_me()] 
	cdouble* m_local; 
	/// object for 1D ffts 
	FFT1D m_fft; 
	/// fft for striding in x 
	FFT1D m_fft_x; 
	/// fft for striding in y 
	FFT1D m_fft_y; 
	/// fft for striding in z 
	FFT1D m_fft_z; 
}; 

#endif