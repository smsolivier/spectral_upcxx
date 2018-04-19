#ifndef _WRITER_H_
#define _WRITER_H_

#include "DataObjects.H"
#include <string> 
#include <vector> 
#include <fstream> 

using namespace std; 

/// Parallel VTK IO class 
class Writer {
public:
	/// constructor. provide base name of output VTK file 
	/** actual name will be name+upcxx::rank_me()_step#.vtk **/ 
	Writer(string name="solution"); 
	/// destructor. cleans up fstream 
	~Writer(); 
	/// add a Scalar to the output list 
	/** stores pointer to memory so output updates with changes **/ 
	void add(Scalar& a_scalar, string a_name); 
	/// add a Vector to the output list 
	void add(Vector& a_vector, string a_name); 
	/// write all variables to VTK 
	void write(); 

	/// set write frequency to not output at every time step 
	/** write every f calls to write **/ 
	void setFreq(int a_f); 
private:
	/// store base name 
	string m_name; 
	/// number of times write has been called 
	int m_count; 
	/// number of files that have been written to file 
	int m_writes; 
	/// store pointers to scalar variables 
	vector<Scalar*> m_scalars; 
	/// names of scalar variables 
	vector<string> m_scalar_names; 
	/// pointers to vector variables 
	vector<Vector*> m_vectors; 
	/// names of vector variables 
	vector<string> m_vector_names; 
	/// output frequency 
	int m_f; 
	ofstream m_out; 
}; 

#endif