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
	/// initialize memory 
	void init(array<INT,DIM> dims, bool physical=true); 
	/// index into the components of the vector 
	Scalar& operator[](int a_i); 
	/// const access to components of vector 
	const Scalar& operator[](int a_i) const; 

	/// component wise transform to fourier space 
	void forward(); 
	/// component wise transform to physical space 
	void inverse();

	/// component wise out of place transform to fourier space 
	void forward(Vector& a_vector) const; 
	/// component wise out of place transform to physical space 
	void inverse(Vector& a_vector) const;  

	/// cross product of two vectors (\f$ (*this) \times a_v\f$) 
	/** does cross product in physical space. returns in fourier space **/ 
	Vector cross(Vector& a_v) const; 
	/// curl \f$ \nabla \times \vec{f} \f$ 
	Vector curl() const; 

	/// Divergence of (*this): \f$ \nabla \cdot \vec{f} \f$ 
	Scalar divergence() const; 
	/// vector laplacian: \f$ \nabla \cdot \nabla \vec{f} \f$ 
	Vector laplacian() const; 

	/// query if (*this) is in fourier space 
	bool isFourier() const; 
	/// query if (*this) is in physical space 
	bool isPhysical() const; 

	/// get full dimensions 
	array<INT,DIM> getDims() const; 
	/// get local parallel truncated start 
	array<INT,DIM> getPStart() const; 
	/// get local parallel truncated end 
	array<INT,DIM> getPEnd() const; 
	/// get parallel truncated dimensions 
	array<INT,DIM> getPDims() const; 
private:
	/// flag (*this) as in fourier space 
	void setFourier(); 
	/// flag (*this) as in physical space 
	void setPhysical(); 
	/// direct access to local data 
	array<cdouble*,DIM> getLocal() const; 

	/// dimensions 
	array<INT,DIM> m_dims; 
	/// total size 
	INT m_N; 
	/// store the DIM scalars 
	array<Scalar,DIM> m_vector; 
}; 

#endif