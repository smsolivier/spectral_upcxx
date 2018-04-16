#include "DataObjects.H" 
#include <iostream>
#ifdef OMP 
#include <omp.h> 
#endif
#include "Timer.H"

// #define TRANSPOSE
// #define SLABS
#define PENCILS

Scalar::Scalar() {
	m_initialized = false; 
}

Scalar::~Scalar() {
	if (m_initialized)
		upcxx::delete_array(m_ptrs[upcxx::rank_me()]); 
}

Scalar::Scalar(array<INT,DIM> N, bool physical) {
	init(N, physical); 
}

void Scalar::init(array<INT,DIM> N, bool physical) {
	m_initialized = true; 
	m_rank = upcxx::rank_me(); 
#ifdef OMP 
	int nthreads; 
	#pragma omp parallel 
	{
		#pragma omp master 
		nthreads = omp_get_num_threads(); 
	}
	if (upcxx::rank_me() == 0) cout << "nthreads = " << nthreads << endl; 
#endif
	m_dims = N; 

	// compute total size 
	m_N = 1; 
	for (INT i=0; i<DIM; i++) {
		m_N *= m_dims[i]; 
	}

	// number of x,y slabs per rank 
	m_Nz = ceil((double)m_dims[2]/upcxx::rank_n()); 
	m_Ny = m_dims[1]/upcxx::rank_n(); 

	m_pdims = {m_dims[0], m_dims[1], m_Nz}; 

	// number of entries per process
	m_dSize = m_Nz*m_dims[0]*m_dims[1]; 

	// store a global pointer for every rank 
	m_ptrs.resize(upcxx::rank_n()); 

	// allocate memory for the slab (truncated in z) 
	m_ptrs[upcxx::rank_me()] = upcxx::new_array<cdouble>(m_dSize); 
	m_local = m_ptrs[upcxx::rank_me()].local(); 

	// broadcast pointer address to other ranks 
	for (INT i=0; i<upcxx::rank_n(); i++) {
		m_ptrs[i] = upcxx::broadcast(m_ptrs[i], i).wait(); 
	}

	// setup FFTW plans 
	Timer setup("setup FFTW plans"); 
	m_fft_x.init(m_dims[0], 1, m_local); 
	m_fft_y.init(m_dims[1], m_dims[0], m_local); 
	// needs to by Ny 
	m_fft_z.init(m_dims[2], m_dims[0]*m_Ny, m_local); 
	setup.stop(); 

	if (physical) setPhysical(); 
	else setFourier(); 
}

void Scalar::operator=(Scalar& scalar) {
	Timer deepcopy("deep copy");  
	init(scalar.getDims(), scalar.isPhysical()); 

	// memcpy local data 
	memcpy(m_local, scalar.getLocal(), m_dSize*sizeof(cdouble)); 
}

void Scalar::set(array<INT,DIM> index, cdouble val) {
	INT rank, loc; 
	getIndex(index, rank, loc); 

	// only rput if not local data 
	if (rank == upcxx::rank_me()) {
		m_local[loc] = val; 
	} else {
		upcxx::rput(val, m_ptrs[rank]+loc).wait(); 
	}
}

cdouble& Scalar::operator[](array<INT,DIM> index) {
	INT rank, loc; 
	getIndex(index, rank, loc); 

	// only do rget if not a local index 
	if (rank == upcxx::rank_me()) {
		return m_local[loc]; 
	} else {
		// return upcxx::rget(m_ptrs[rank]+loc).wait(); 
		cerr << "ERROR (Scalar.cpp): accessing non-local index" << endl; 
		upcxx::finalize(); 
		exit(0); 
	}
}

cdouble& Scalar::operator[](INT index) {
	if (index >= m_dSize) {
		cout << "ERROR (Scalar.cpp): index out of range in direct access operator" << endl; 
	}
	return m_local[index]; 
}

array<double,DIM> Scalar::freq(array<INT,DIM> ind) {
	array<double,DIM> k; 
	for (int i=0; i<DIM; i++) {
		if (ind[i] <= m_dims[i]/2) k[i] = ind[i]; 
		else k[i] = -m_dims[i] + ind[i]; 
	}

	return k; 
}

void Scalar::forward() {
	Timer forward("forward transform"); 
	if (isFourier()) {
		cout << "ERROR (Scalar.cpp): already in fourier space" << endl; 
		upcxx::finalize(); 
		exit(0); 
	}
	transform(1); 
	setFourier(); 
}

void Scalar::inverse() {
	Timer inverse("inverse transform"); 
	if (isPhysical()) {
		cout << "ERROR (Scalar.cpp): already in physical space" << endl; 
		upcxx::finalize(); 
		exit(0); 
	}
	transform(-1); 

	// divide by N^3 
	#pragma omp parallel for 
	for (INT i=0; i<m_dSize; i++) {
		m_local[i] /= m_N; 
	}
	setPhysical(); 
}

void Scalar::forward(Scalar& a_scalar) {
	// deep copy 
	a_scalar = (*this); 

	a_scalar.forward(); 
}

void Scalar::inverse(Scalar& a_scalar) {
	// deep copy 
	a_scalar = (*this); 

	a_scalar.inverse(); 
}

void Scalar::add(Scalar& a) {
	#pragma omp parallel for schedule(static)
	for (INT i=0; i<m_dSize; i++) {
		m_local[i] += a[i];  
	}
}

Vector Scalar::gradient() {
	Timer grad_timer("scalar gradient"); 

	if (!isFourier()) {
		cerr << "ERROR (Scalar.cpp): scalar must be in fourier space for gradient" << endl; 
		upcxx::finalize(); 
		exit(0); 
	}

	// return a vector in fourier space 
	Vector v(m_dims, false); 
	cdouble imag(0,1.); 
	array<INT,DIM> ind = {0,0,0}; 
	// #pragma omp parallel for 
	for (ind[2]=m_rank*m_Nz; ind[2]<(m_rank+1)*m_Nz; ind[2]++) {
		for (ind[1]=0; ind[1]<m_pdims[1]; ind[1]++) {
			for (ind[0]=0; ind[0]<m_pdims[0]; ind[0]++) {
				array<double,DIM> k = freq(ind); 
				for (int i=0; i<DIM; i++) {
					v[i][ind] = imag*k[i]*(*this)[ind]; 
				}
			}
		}
	}
	return v; 
}

Scalar Scalar::laplacian() {
	Timer lap_timer("scalar laplacian"); 

	if (!isFourier()) {
		cerr << "ERROR (Scalar.cpp): scalar must be in fourier space for laplacian" << endl; 
		upcxx::finalize(); 
		exit(0); 
	}

	// return scalar in fourier space 
	Scalar lap(m_dims, false); 
	array<INT,DIM> ind = {0,0,0}; 
	array<double,DIM> k; 
	for (ind[2]=m_rank*m_Nz; ind[2]<(m_rank+1)*m_Nz; ind[2]++) {
		for (ind[1]=0; ind[1]<m_pdims[1]; ind[1]++) {
			for (ind[0]=0; ind[0]<m_pdims[0]; ind[0]++) {
				k = freq(ind); 
				lap[ind] = -(k[0]*k[0]+k[1]*k[1]+k[2]*k[2])*
					(*this)[ind]; 
			}
		}
	}
	return lap; 
}

void Scalar::zeroHighModes() {
	array<INT,DIM> half = m_dims; 
	for (int i=0; i<DIM; i++) {
		half[i] /= 2; 
	}

	// zero out x 
	array<INT,DIM> ind = {half[0],0,0}; 
	for (ind[1]=0; ind[1]<m_dims[1]; ind[1]++) {
		for (ind[2]=m_rank*m_Nz; ind[2]<(m_rank+1)*m_Nz; ind[2]++) {
			(*this)[ind] = 0; 
		}
	}

	// zero out y 
	ind = {0,half[1],0}; 
	for (ind[2]=m_rank*m_Nz; ind[2]<(m_rank+1)*m_Nz; ind[2]++) {
		for (ind[0]=0; ind[0]<m_dims[0]; ind[0]++) {
			(*this)[ind] = 0; 
		}
	}

	// zero out z 
	ind = {0,0,half[2]}; 
	for (ind[1]=0; ind[1]<m_dims[1]; ind[1]++) {
		for (ind[0]=0; ind[0]<m_dims[0]; ind[0]++) {
			(*this)[ind] = 0; 
		}
	}
}

INT Scalar::localSize() {return m_dSize; } 
array<INT,DIM> Scalar::getDims() {return m_dims; }
array<INT,DIM> Scalar::getPDims() {return m_pdims; }
array<INT,DIM> Scalar::getPStart() {
	array<INT,DIM> start = {0,0,upcxx::rank_me()*m_Nz}; 
	return start; 
}
array<INT,DIM> Scalar::getPEnd() {
	array<INT,DIM> end = {m_dims[0], m_dims[1], (upcxx::rank_me()+1)*m_Nz}; 
	return end; 
}
INT Scalar::size() {return m_N; } 
cdouble* Scalar::getLocal() {return m_local; } 
double Scalar::memory() {
	if (upcxx::rank_me() == 0) {
		cout << "memory requirement = " << (double)m_N*sizeof(cdouble)/1e9 << " GB" << endl; 
	}
}

void Scalar::setPhysical() {m_fourier = false; }
void Scalar::setFourier() {m_fourier = true; }
bool Scalar::isPhysical() {return !m_fourier; }
bool Scalar::isFourier() {return m_fourier; } 

void Scalar::transform(int DIR) {
	// store transposed data in tmp 
	vector<upcxx::global_ptr<cdouble>> tmp(upcxx::rank_n(), NULL);
	// allocate global memory 
	tmp[upcxx::rank_me()] = upcxx::new_array<cdouble>(m_dSize);
	// broadcast to other ranks 
	for (int i=0; i<upcxx::rank_n(); i++) {
		tmp[i] = upcxx::broadcast(tmp[i], i).wait(); 
	}
	// get local access 
	cdouble* tlocal = tmp[upcxx::rank_me()].local(); 

	// --- transform columns --- 
	Timer cols("columns"); 
#ifdef OMP 
	#pragma omp parallel 
	{
		#pragma omp for 
		for (INT k=0; k<m_Nz; k++) {
			for (INT i=0; i<m_dims[0]; i++) {
				cdouble* start = m_local + i + k*m_dims[0]*m_dims[1]; 
				m_fft_y.transform(start, DIR); 
			}
		}
	}
#else
	for (INT k=0; k<m_Nz; k++) {
		for (INT i=0; i<m_dims[0]; i++) {
			cdouble* start = m_local + i + k*m_dims[0]*m_dims[1]; 
			// m_fft.transform(start, 
				// m_dims[1], m_dims[0], DIR); 
			m_fft_y.transform(start, DIR); 
		}
	}
#endif
#ifdef VERBOSE
	cout << "columns" << endl; 
#endif
	cols.stop(); 
	upcxx::barrier(); 

	// --- transforms rows and pencil transpose --- 
	Timer rows("rows"); 
	INT Ny = m_dims[1]/upcxx::rank_n(); 
#if defined SLABS & defined OMP 
	// do each face in parallel and then serially send the whole face off (message size = Nx * Ny/p)
	for (INT k=0; k<m_Nz; k++) {
		#pragma omp parallel for 
		for (INT j=0; j<m_dims[1]; j++) {
			cdouble* start = m_local + j*m_dims[0] + k*m_dims[0]*m_dims[1]; 
			m_fft_x.transform(start, DIR); 
		}

		// send slabs  
		for (int j=0; j<upcxx::rank_n(); j++) {
			cdouble* start = m_local + k*m_dims[0]*m_dims[1]; 
			INT loc = upcxx::rank_me()*m_Ny*m_dims[0]*m_Nz + k*m_dims[0]*m_Ny; 
			if (j == upcxx::rank_me()) {
				memcpy(tlocal+loc, start, m_dims[0]*m_Ny*sizeof(cdouble)); 
			} else {
				upcxx::rput(start, tmp[j]+loc, m_dims[0]*m_Ny);			
			}
		}
	}
#elif defined PENCILS & defined OMP 
	#pragma omp parallel
	{
		upcxx::future<> fut = upcxx::make_future(); 
		for (INT k=0; k<m_Nz; k++) {
			for (INT j=0; j<m_dims[1]; j++) {
				cdouble* row = m_local + j*m_dims[0] + k*m_dims[0]*m_dims[1]; 
				m_fft_x.transform(row, DIR); 
				INT send_to = j/Ny; 
				INT row_num = j % Ny;
				INT loc = upcxx::rank_me()*m_Nz*m_dims[0]*Ny + 
					k*m_dims[0]*Ny + 
					m_dims[0]*row_num; 
				if (loc >= m_N) cout << "out of range" << endl; 
				if (send_to == upcxx::rank_me()) {
					memcpy(tlocal+loc, row, m_dims[0]*sizeof(cdouble)); 
				} else {
					fut = upcxx::when_all(fut, 
						upcxx::rput(row, tmp[send_to]+loc, m_dims[0])); 
				}
			}
		}
		fut.wait();
	}
#else
	upcxx::future<> fut = upcxx::make_future(); 
	for (INT k=0; k<m_Nz; k++) {
		for (INT j=0; j<m_dims[1]; j++) {
			cdouble* row = m_local + j*m_dims[0] + k*m_dims[0]*m_dims[1]; 
			m_fft_x.transform(row, DIR); 
			INT send_to = j/Ny; 
			INT row_num = j % Ny;
			INT loc = upcxx::rank_me()*m_Nz*m_dims[0]*Ny + 
				k*m_dims[0]*Ny + 
				m_dims[0]*row_num; 
			if (loc >= m_N) cout << "out of range" << endl; 
			if (send_to == upcxx::rank_me()) {
				memcpy(tlocal+loc, row, m_dims[0]*sizeof(cdouble)); 
			} else {
				fut = upcxx::when_all(fut, 
					upcxx::rput(row, tmp[send_to]+loc, m_dims[0])); 
			}
		}
	}
	fut.wait(); 
#endif
	rows.stop(); 
	upcxx::barrier(); 
#ifdef VERBOSE
	cout << "rows" << endl; 
#endif

	// --- transform in z direction --- 
	Timer ztimer("z rows"); 
#ifdef TRANSPOSE 
	transposeX2Z(tlocal); 
#endif
#ifdef OMP 
	#pragma omp parallel 
	{
		#pragma omp for 
		for (INT j=0; j<Ny; j++) {
			for (INT i=0; i<m_dims[0]; i++) {
				cdouble* start = tlocal + i + j*m_dims[0]; 
				#ifdef TRANSPOSE
					m_fft_x.transform(start, DIR);
				#else
					m_fft_z.transform(start, DIR); 
				#endif
			}
		}
	}
#else 
	for (INT j=0; j<Ny; j++) {
		for (INT i=0; i<m_dims[0]; i++) {
			cdouble* zrow = tlocal + i + j*m_dims[0]; 
			m_fft_z.transform(zrow, DIR); 
		}
	}
#endif
#ifdef TRANSPOSE
	transposeZ2X(tlocal); 
#endif
	ztimer.stop(); 
	upcxx::barrier(); 
#ifdef VERBOSE
	cout << "z" << endl; 
#endif

	// switch back to original data layout (send slabs of Nx*Ny) 
	Timer trans("transpose back"); 
	upcxx::future<> f = upcxx::make_future(); 
	for (INT k=0; k<m_dims[2]; k++) {
		cdouble* slab = tlocal + k*Ny*m_dims[0]; 
		INT send_to = k/m_Nz; 
		INT rem = k % m_Nz; 
		INT dest = upcxx::rank_me()*Ny*m_dims[0] + 
			rem*m_dims[0]*m_dims[1]; 
		if (send_to == upcxx::rank_me()) {
			memcpy(m_local+dest, slab, m_dims[0]*Ny*sizeof(cdouble)); 
		} else {
			f = upcxx::when_all(f, 
				upcxx::rput(slab, m_ptrs[send_to]+dest, m_dims[0]*Ny)); 
		}
	}
	f.wait(); 
	trans.stop(); 
#ifdef VERBOSE 
	cout << "transpose back" << endl; 
#endif

	// clean up temp variable 
	upcxx::barrier(); 
	upcxx::delete_array(tmp[upcxx::rank_me()]);
#ifdef VERBOSE
	cout << "completed transform in " << DIR << " direction" << endl; 
#endif
}

void Scalar::transposeX2Z(cdouble* f) {
	/* assumes truncated direction is in y ie after global transpose step */ 
	cdouble* tmp = new cdouble[m_dSize];
	memcpy(tmp, f, m_dSize*sizeof(cdouble)); 
	#pragma omp parallel for 
	for (int i=0; i<m_dims[0]; i++) {
		for (int j=0; j<m_Ny; j++) {
			for (int k=0; k<m_dims[2]; k++) {
				f[k+m_dims[2]*j+m_dims[2]*m_Ny*i] = 
					tmp[i+j*m_dims[0]+k*m_dims[0]*m_Ny]; 
			}
		}
	}
	delete(tmp); 
}

void Scalar::transposeZ2X(cdouble* f) {
	/* assumes truncated direction is in y ie after global transpose step */ 
	cdouble* tmp = new cdouble[m_dSize];
	memcpy(tmp, f, m_dSize*sizeof(cdouble)); 
	#pragma omp parallel for 
	for (int i=0; i<m_dims[0]; i++) {
		for (int j=0; j<m_Ny; j++) {
			for (int k=0; k<m_dims[2]; k++) {
				f[i+j*m_dims[0]+k*m_dims[0]*m_Ny] = 
					tmp[k+m_dims[2]*j+m_dims[2]*m_Ny*i]; 
			}
		}
	}
	delete(tmp); 
}

void Scalar::getIndex(array<INT,DIM> index, INT& rank, INT& loc) {
	INT n = index[0] + m_dims[0]*index[1] + index[2]*m_dims[0]*m_dims[1]; 
	if (n >= m_N) {
		cout << "ERROR: index out of range in Scalar.cpp" << endl; 
		cout << "n = " << n << endl; 
		upcxx::finalize(); // close upcxx 
		exit(0); // exit program 
	}
	rank = n/m_dSize; // which rank owns the data 
	loc = n % m_dSize; // remainder is index into the local array 
}
