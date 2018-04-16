#include "Writer.H"
#include "Timer.H"
#include <upcxx/upcxx.hpp> 
#include "VisitWriter.H"

Writer::Writer(string name) {
	m_name = name; 
	m_count = 0; 
	m_writes = 0; 
	m_f = 1; 
	if (upcxx::rank_me() == 0) {
		m_out.open(name+".visit"); 
		m_out << "!NBLOCKS " << upcxx::rank_n() << endl; 
	}
}

Writer::~Writer() {
	if (upcxx::rank_me() == 0) m_out.close(); 
}

void Writer::add(Scalar& a_scalar, string a_name) {
	m_scalars.push_back(&a_scalar); 
	m_scalar_names.push_back(a_name); 
}

void Writer::add(Vector& a_vector, string a_name) {
	m_vectors.push_back(&a_vector); 
	m_vector_names.push_back(a_name); 
}

void Writer::setFreq(int a_f) {m_f = a_f; } 

void Writer::write() {
	Timer timer_write("write to VTK"); 

	if (m_count++%m_f != 0) {
		return; 
	}

	int mrank = upcxx::rank_me(); 

	int nvars = m_scalars.size() + m_vectors.size(); 
	int n_scalars = m_scalars.size(); 
	int n_vectors = m_vectors.size(); 

	array<INT,DIM> dims; 
	array<INT,DIM> pdims; 
	if (n_scalars > 0) {
		dims = m_scalars[0]->getDims(); 
		pdims = m_scalars[0]->getPDims(); 
	} else if (n_vectors > 0) {
		dims = m_vectors[0]->getDims(); 
		pdims = m_vectors[0]->getPDims(); 
	} else {
		cerr << "ERROR (Writer.cpp): variables list is empty" << endl; 
		upcxx::finalize(); 
		exit(0); 
	}

	// copy if not in physical space 
	vector<Scalar> scalars(n_scalars); 
	vector<Vector> vectors(n_vectors); 
	// perform inverse if necessary 
	for (int i=0; i<n_scalars; i++) {
		if (m_scalars[i]->isFourier()) {
			m_scalars[i]->inverse(scalars[i]); // out of place transform 
		} else {
			scalars[i] = *m_scalars[i]; // copy pointer over 
		}
	}
	for (int i=0; i<n_vectors; i++) {
		if (m_vectors[i]->isFourier()) {
			m_vectors[i]->inverse(vectors[i]); 
		} else {
			vectors[i] = *m_vectors[i]; 
		}
	}

	// deep copy to float** 
	float** vars = new float*[nvars]; 
	for (int i=0; i<n_scalars; i++) {
		vars[i] = new float[scalars[i].localSize()]; 
		for (int j=0; j<scalars[i].localSize(); j++) {
			vars[i][j] = scalars[i][j].real(); 
		}
	}

	// vector version 
	for (int i=0; i<n_vectors; i++) {
		int vind = i + n_scalars; 
		vars[vind] = new float[vectors[0][0].localSize()*DIM]; 
		for (int j=0; j<vectors[0][0].localSize(); j++) {
			for (int d=0; d<DIM; d++) {
				vars[vind][DIM*j+d] = vectors[i][d][j].real(); 
			}
		}
	}

	// number of dimensions for each variable 
	int vardim[nvars]; 
	for (int i=0; i<n_scalars; i++) {
		vardim[i] = 1; 
	}
	for (int i=n_scalars; i<nvars; i++) {
		vardim[i] = DIM; 
	}

	// centering of the variables 
	int centering[nvars]; 
	for (int i=00; i<nvars; i++) {
		centering[i] = 1; 
	}

	// variable names 
	const char* varnames[nvars]; 
	for (int i=0; i<n_scalars; i++) {
		varnames[i] = m_scalar_names[i].c_str(); 
	}
	for (int i=n_scalars; i<nvars; i++) {
		varnames[i] = m_vector_names[i-n_scalars].c_str(); 
	}

	// grid locations 
	vector<float> x(pdims[0]); 
	vector<float> y(pdims[1]); 
	vector<float> z(pdims[2]); 

	double xb = 2*M_PI; 
	double yb = 2*M_PI; 
	double zb = 2*M_PI; 
	for (int i=0; i<pdims[0]; i++) {
		x[i] = i*xb/dims[0]; 
	}
	for (int i=0; i<pdims[1]; i++) {
		y[i] = i*yb/dims[0]; 
	}
	for (int i=mrank*pdims[2]; i<(mrank+1)*pdims[2]; i++) {
		z[i-mrank*pdims[2]] = i*zb/dims[2]; 
	}

	// append to master file 
	if (upcxx::rank_me() == 0) {
		for (int i=0; i<upcxx::rank_n(); i++) {
			m_out << m_name << i << "_" << m_writes << ".vtk" << endl; 
		}
	}

	string fname = m_name + to_string(upcxx::rank_me())
		+ "_" + to_string(m_writes++); 
	// convert to normal int 
	array<int,DIM> sdims; 
	for (int i=0; i<DIM; i++) {
		sdims[i] = (int)pdims[i]; 
	}

	write_rectilinear_mesh(fname.c_str(), VISIT_ASCII, &sdims[0], 
		&x[0], &y[0], &z[0], nvars, &vardim[0], &centering[0], varnames, vars);

	// clean up pointers 
	for (int i=0; i<nvars; i++) {
		delete[] vars[i]; 
	}

	delete vars; 
}