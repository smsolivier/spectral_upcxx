#ifndef _OPERATORS_H_
#define _OPERATORS_H_

/// element wise scalar * scalar 
Scalar operator*(const Scalar& a, const Scalar& b); 
/// element wise scalar - scalar 
Scalar operator-(const Scalar& a, const Scalar& b); 
/// element wise scalar + scalar 
Scalar operator+(const Scalar& a, const Scalar& b); 
/// element wise Vector - Vector 
Vector operator-(const Vector& a, const Vector& b); 
/// element wise Vector + Vector 
Vector operator+(const Vector& a, const Vector& b); 
/// scale s by alpha 
Scalar operator*(double alpha, const Scalar& s); 
/// scale v by alpha 
Vector operator*(double alpha, const Vector& v); 

#endif