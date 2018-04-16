#ifndef _VECTOR_H_
#define	_VECTOR_H_

/// Represent a vector with scalars in DIM directions 
class Vector {
public:
	/// default constructor 
	Vector(); 
	/// constructor 
	/** supply Nx, Ny, Nz. Defaults to building in physical space **/ 
	Vector(array<INT,DIM> dims, bool physical=true); 
	/// index into the components of the vector 
	Scalar& operator[](int a_i); 
	/// copy constructor 
	void operator=(Vector& a_v); 

	/// component wise transform to fourier space 
	void forward(); 
	/// component wise transform to physical space 
	void inverse();

	/// component wise out of place transform to fourier space 
	void forward(Vector& a_vector); 
	/// component wise out of place transform to physical space 
	void inverse(Vector& a_vector);  

	/// cross product of two vectors (\f$ (*this) \times a_v\f$) 
	/** does cross product in physical space. returns in fourier space **/ 
	Vector cross(Vector& a_v); 
	/// curl \f$ \nabla \times \vec{f} \f$ 
	Vector curl(); 

	/// Divergence of (*this): \f$ \nabla \cdot \vec{f} \f$ 
	Scalar divergence(); 
	/// vector laplacian: \f$ \nabla \cdot \nabla \vec{f} \f$ 
	Vector laplacian(); 

	/// query if (*this) is in fourier space 
	bool isFourier(); 
	/// query if (*this) is in physical space 
	bool isPhysical(); 

	/// direct access to local data 
	array<cdouble*,DIM> getLocal(); 
	/// get full dimensions 
	array<INT,DIM> getDims(); 
	/// get local parallel truncated start 
	array<INT,DIM> getPStart(); 
	/// get local parallel truncated end 
	array<INT,DIM> getPEnd(); 
	/// get parallel truncated dimensions 
	array<INT,DIM> getPDims(); 
private:
	/// flag (*this) as in fourier space 
	void setFourier(); 
	/// flag (*this) as in physical space 
	void setPhysical(); 

	/// dimensions 
	array<INT,DIM> m_dims; 
	/// total size 
	INT m_N; 
	/// store the DIM scalars 
	array<Scalar,DIM> m_vector; 
}; 

#endif