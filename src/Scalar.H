#ifndef __SCALAR_H__ 
#define __SCALAR_H__

#include "FFT1D.H"

/// Store a 3D array with UPCXX 
/** stores x,y locally and distributes in z **/ 
class Scalar {
public: 
	/// default constructor 
	Scalar(); 
	/// constructor 
	/** set size in DIM directions. Defaults to building in physical space **/ 
	Scalar(array<INT,DIM> N, bool physical=true); 
	/// destructor 
	~Scalar(); 
	/// copy constructor 
	Scalar(const Scalar& scalar); 
	/// copy assignment. Does a deep copy 
	void operator=(const Scalar& scalar); 
	/// initialize the array after calling default constructor 
	void init(array<INT,DIM> N, bool physical=true); 
	/// set a value in the distributed array 
	void set(array<INT,DIM> index, cdouble val); 
	/// get a value from the distributed array 
	cdouble get(array<INT,DIM> index) const; 
	/// get a local element from the distributed array 
	/** can only access local values **/ 
	cdouble& operator[](array<INT,DIM> index); 
	/// const access to distributed array 
	/** can only access local values **/ 
	cdouble operator[](array<INT,DIM> index) const; 
	/// get a local element from the distributed array (with 3D indexing) 
	cdouble& operator()(INT a_i, INT a_j, INT a_k); 
	/// const access to a local element (with 3D indexing) 
	cdouble operator()(INT a_i, INT a_j, INT a_k) const; 
	/// direct access to local pointer 
	cdouble& operator[](INT index); 
	/// const access to local pointer 
	cdouble operator[](INT index) const; 
	/// return frequency for grid point ind 
	array<double,DIM> freq(array<INT,DIM> ind) const; 

	/// in place forward transform 
	void forward(); 
	/// in place reverse transform 
	void inverse();

	/// out of place forward transform 
	void forward(Scalar& a_scalar) const; 
	/// out of place reverse transform 
	void inverse(Scalar& a_scalar) const; 

	/// compute gradient \f$ \nabla f \f$. Returns in fourier space 
	Vector gradient() const; 
	/// compute laplacian: \f$ \nabla^2 f \f$. Returns in fourier space 
	Scalar laplacian() const; 

	/// invert laplacian 
	/** inverts \f$ a + b\nabla^2 \f$ where a, b are scalars\n 
		Divides by: \f$ a - b(m^2 + n^2 + p^2) \f$ **/ 
	void laplacian_inverse(double a=0, double b=1); 

	/// zero out the highest modes 
	void zeroHighModes(); 

	/// get size per processor 
	INT localSize() const; 
	/// get dimensions 
	array<INT,DIM> getDims() const; 
	/// get parallel truncated dimensions 
	array<INT,DIM> getPDims() const; 
	/// get beginning of local truncated dimensions 
	array<INT,DIM> getPStart() const; 
	/// get end of truncated local dimensions
	array<INT,DIM> getPEnd() const; 
	/// get total size 
	INT size() const; 
	/// compute allocated memory size per processor 
	double memory() const; 

	/// query if (*this) is in physical space 
	bool isPhysical() const; 
	/// query if (*this) is in fourier space 
	bool isFourier() const; 
	/// flag (*this) as in fourier space 
	void setFourier(); 
	/// flag (*this) as in physical space 
	void setPhysical(); 

	/// get the average value of the data 
	double average() const; 
private:
	/// perform fourier transform in parallel 
	void transform(int dir); 
	/// transpose so that contiguous dimension is in z  
	void transposeX2Z(cdouble* f); 
	/// transpose from contiguous in z to contiguous in x 
	void transposeZ2X(cdouble* f); 
	/// get the rank and location into the distributed array 
	inline void getIndex(array<INT,DIM> index, INT& rank, INT& loc) const; 
	/// get pointer to local data
	cdouble* getLocal() const; 

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
	/// parallel truncated dimensions 
	array<INT,DIM> m_pdims; 
	/// global pointers to z slabs 
	vector<upcxx::global_ptr<cdouble>> m_ptrs; 
	/// local version of m_ptrs[rank_me()] 
	cdouble* m_local; 
	/// fft for striding in x 
	FFT1D m_fft_x; 
	/// fft for striding in y 
	FFT1D m_fft_y; 
	/// fft for striding in z 
	FFT1D m_fft_z; 
	/// store upcxx rank 
	int m_rank; 

	/// store if in physical/fourier space 
	bool m_fourier; 

	/// store if memory is allocated with init 
	bool m_initialized; 

	/// total scalars created 
	static int m_nscalars; 
}; 

#endif