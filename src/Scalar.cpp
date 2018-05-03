#include "DataObjects.H" 
#include <iostream>
#ifdef OMP 
#include <omp.h> 
#endif
#include "CH_Timer.H"
#include <cassert> 

#define ERROR(message) \
	cout << "ERROR in " << __func__ << " (" << __FILE__ \
		<< " " << __LINE__ << ", r" << upcxx::rank_me() << "): " << message << endl; \
	upcxx::finalize(); \
	exit(0);

double Scalar::average() const {
	cdouble avg = 0; 
	for (int i=0; i<localSize(); i++) {
		avg += abs((*this)[i]); 
	}

	return avg.real()/(double)localSize(); 
}

Scalar::Scalar() {
	m_initialized = false; 
	m_ptrs.resize(upcxx::rank_n()); 
	m_ptrs[upcxx::rank_me()] = nullptr; 
	m_local = nullptr; 
	m_rank = upcxx::rank_me(); 
}

Scalar::Scalar(array<INT,DIM> N, bool physical) {
	init(N, physical); 
}

Scalar::~Scalar() {
	if (m_initialized) {
		upcxx::delete_array(m_ptrs[upcxx::rank_me()]);

		m_ptrs[upcxx::rank_me()] = nullptr; 
		m_local = nullptr; 

		m_nscalars--; 
		m_initialized = false; 
	}
}

Scalar::Scalar(const Scalar& scalar) {
	init(scalar.getDims(), scalar.isPhysical()); 

	// memcpy local data 
	memcpy(m_local, scalar.getLocal(), m_dSize*sizeof(cdouble));
}

void Scalar::operator=(const Scalar& scalar) {
	if (!m_initialized) {
		init(scalar.getDims(), scalar.isPhysical()); 
	}

	// memcpy local data 
	memcpy(m_local, scalar.getLocal(), m_dSize*sizeof(cdouble)); 
}

void Scalar::init(array<INT,DIM> N, bool physical) {
	m_initialized = true; 
	m_nscalars++; 

	m_rank = upcxx::rank_me(); 

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

	m_ptrs.resize(upcxx::rank_n()); 

	// allocate memory for the slab (truncated in z) 
	m_ptrs[upcxx::rank_me()] = upcxx::new_array<cdouble>(m_dSize); 
	m_local = m_ptrs[upcxx::rank_me()].local(); 

	// broadcast pointer address to other ranks 
	for (INT i=0; i<upcxx::rank_n(); i++) {
		m_ptrs[i] = upcxx::broadcast(m_ptrs[i], i).wait(); 
	}
	upcxx::barrier(); 

	// setup FFTW plans 
	m_fft_x.init(m_dims[0], 1, m_local); 
	m_fft_y.init(m_dims[1], m_dims[0], m_local); 
	// needs to by Ny 
	m_fft_z.init(m_dims[2], m_dims[0]*m_Ny, m_local); 

	if (physical) setPhysical(); 
	else setFourier(); 

	upcxx::barrier(); 
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

cdouble Scalar::get(array<INT,DIM> index) const {
	INT rank, loc; 
	getIndex(index, rank, loc); 

	// only do rget if not a local index 
	if (rank == upcxx::rank_me()) {
		return m_local[loc]; 
	} else {
		return upcxx::rget(m_ptrs[rank]+loc).wait(); 
	}
}

cdouble& Scalar::operator[](array<INT,DIM> index) {
	INT rank, loc; 
	getIndex(index, rank, loc); 

	// only do rget if not a local index 
	if (rank == upcxx::rank_me()) {
		return m_local[loc]; 
	} else {
		ERROR("accessed non-local index use set/get"); 
	}
}

cdouble Scalar::operator[](array<INT,DIM> index) const {
	INT rank, loc; 
	getIndex(index, rank, loc); 

	// only do rget if not a local index 
	if (rank == upcxx::rank_me()) {
		return m_local[loc]; 
	} else {
		ERROR("accessed non-local index use get"); 
	}
}

cdouble& Scalar::operator()(INT a_i, INT a_j, INT a_k) {
	array<INT,DIM> index = {a_i, a_j, a_k}; 
	return (*this)[index]; 
}

cdouble Scalar::operator()(INT a_i, INT a_j, INT a_k) const {
	array<INT,DIM> index = {a_i, a_j, a_k}; 
	return (*this)[index]; 
}
cdouble& Scalar::operator[](INT index) {
	if (index >= m_dSize) {
		ERROR("accessed out of local data"); 
	}
	return m_local[index]; 
}

cdouble Scalar::operator[](INT index) const {
	if (index >= m_dSize) {
		ERROR("accessed out of local data"); 
	}
	return m_local[index]; 
}

array<double,DIM> Scalar::freq(array<INT,DIM> ind) const {
	array<double,DIM> k; 
	for (int i=0; i<DIM; i++) {
		if (ind[i] <= m_dims[i]/2) k[i] = (double)ind[i]; 
		else k[i] = -1.*(double)m_dims[i] + (double)ind[i];
	}

	return k; 
}

void Scalar::forward() {
	if (isFourier()) {
		ERROR("already in fourier space"); 
	}
	transform(-1); 
	zeroHighModes(); 
	setFourier(); 
}

void Scalar::inverse() {
	if (isPhysical()) {
		ERROR("already in physical space"); 
	}
	transform(1); 

	// divide by N^3 
	#pragma omp parallel for 
	for (INT i=0; i<m_dSize; i++) {
		m_local[i] /= m_N; 
	}
	setPhysical(); 
}

void Scalar::forward(Scalar& a_scalar) const {
	// deep copy 
	a_scalar = (*this); 

	a_scalar.forward(); 
}

void Scalar::inverse(Scalar& a_scalar) const {
	// deep copy 
	a_scalar = (*this); 

	a_scalar.inverse(); 
}

Vector Scalar::gradient() const {
	CH_TIMERS("gradient"); 
	if (!isFourier()) {
		ERROR("must begin gradient in fourier space"); 
	}

	// return a vector in fourier space 
	Vector v(m_dims, false); 

	#pragma omp parallel 
	{
		cdouble imag(0,1.); 
		array<INT,DIM> ind = {0,0,0}; 
		array<INT,DIM> start = getPStart(); 
		array<INT,DIM> end = getPEnd(); 
		array<double,DIM> f; 

		#pragma omp for 
		for (INT k=start[2]; k<end[2]; k++) {
			ind[2] = k; 
			for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
				for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
					f = freq(ind); 
					for (int i=0; i<DIM; i++) {
						v[i][ind] = imag*f[i]*(*this)[ind]; 
					}
				}
			}
		}
	}
	return v; 
}

Scalar Scalar::laplacian() const {
	CH_TIMERS("scalar laplacian"); 
	if (!isFourier()) {
		ERROR("must begin laplacian in fourier space"); 
	}

	// return scalar in fourier space 
	Scalar lap(m_dims, false); 

	#pragma omp parallel 
	{
		array<INT,DIM> ind = {0,0,0}; 
		array<INT,DIM> start = getPStart(); 
		array<INT,DIM> end = getPEnd(); 
		array<double,DIM> f; 
		#pragma omp for 
		for (INT k=start[2]; k<end[2]; k++) {
			for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
				for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
					ind[2] = k; 
					f = freq(ind); 
					lap[ind] = -(f[0]*f[0]+f[1]*f[1]+f[2]*f[2])*
						(*this)[ind]; 
				}
			}
		}
	}
	return lap; 
}

void Scalar::laplacian_inverse(double a, double b) {
	CH_TIMERS("laplacian inverse"); 
	
	#pragma omp parallel 
	{
		array<INT,DIM> start = getPStart(); 
		array<INT,DIM> end = getPEnd(); 
		array<INT,DIM> ind; 
		array<double,DIM> f; 
		#pragma omp for 
		for (INT k=start[2]; k<end[2]; k++) {
			for (ind[1]=start[1]; ind[1]<end[1]; ind[1]++) {
				for (ind[0]=start[0]; ind[0]<end[0]; ind[0]++) {
					ind[2] = k; 
					f = freq(ind); 
					double sum = 0; 
					for (int d=0; d<DIM; d++) {
						sum += f[d]*f[d]; 
					}
					if (sum != 0) {
						(*this)[ind] /= (a - b*sum); 
					}
				}
			}
		}
	}
}

void Scalar::zeroHighModes() {
#ifdef ZERO
	array<INT,DIM> half; 
	for (int i=0; i<DIM; i++) {
		half[i] = m_dims[i]/2; 
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
	INT rank, loc; 
	getIndex(ind, rank, loc); 
	if (rank != upcxx::rank_me()) return; 
	for (ind[1]=0; ind[1]<m_dims[1]; ind[1]++) {
		for (ind[0]=0; ind[0]<m_dims[0]; ind[0]++) {
			(*this)[ind] = 0; 
		}
	}
#endif
}

INT Scalar::localSize() const {return m_dSize; } 
array<INT,DIM> Scalar::getDims() const {return m_dims; }
array<INT,DIM> Scalar::getPDims() const {return m_pdims; }
array<INT,DIM> Scalar::getPStart() const {
	array<INT,DIM> start = {0,0,upcxx::rank_me()*m_Nz}; 
	return start; 
}
array<INT,DIM> Scalar::getPEnd() const {
	array<INT,DIM> end = {m_dims[0], m_dims[1], (upcxx::rank_me()+1)*m_Nz}; 
	return end; 
}
INT Scalar::size() const {return m_N; } 
cdouble* Scalar::getLocal() const {return m_local; } 
double Scalar::memory() const {
	if (upcxx::rank_me() == 0) {
		cout << "memory requirement = " << 
			(double)m_nscalars*(double)m_N*sizeof(cdouble)/1e9 << " GB" << endl; 
	}
}

void Scalar::setPhysical() {m_fourier = false; }
void Scalar::setFourier() {m_fourier = true; }
bool Scalar::isPhysical() const {return !m_fourier; }
bool Scalar::isFourier() const {return m_fourier; } 

void Scalar::transform(int DIR) {
	CH_TIMERS("transform"); 
	CH_TIMER("setup tmp", ttmp); 
	CH_START(ttmp); 
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
	upcxx::barrier(); 
	CH_STOP(ttmp); 

	// --- transform columns --- 
#ifdef OMP 
	CH_TIMER("columns (OMP)", col); 
	CH_START(col); 
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
	CH_STOP(col); 
#else
	CH_TIMER("columns (serial)", col); 
	CH_START(col); 
	for (INT k=0; k<m_Nz; k++) {
		for (INT i=0; i<m_dims[0]; i++) {
			cdouble* start = m_local + i + k*m_dims[0]*m_dims[1]; 
			m_fft_y.transform(start, DIR); 
		}
	}
	CH_STOP(col); 
#endif

	// upcxx::barrier(); 

	// --- transforms rows and pencil transpose --- 
	INT Ny = m_dims[1]/upcxx::rank_n(); 
#if defined SLABS
	CH_TIMER("rows (slabs)", row); 
	CH_START(row); 
	// do each face in parallel and then serially send the whole face off (message size = Nx * Ny/p)
	upcxx::future<> fut = upcxx::make_future(); 
	for (INT k=0; k<m_Nz; k++) {
		#pragma omp parallel for 
		for (INT j=0; j<m_dims[1]; j++) {
			cdouble* start = m_local + j*m_dims[0] + k*m_dims[0]*m_dims[1]; 
			m_fft_x.transform(start, DIR); 
		}

		// send slabs  
		for (int j=0; j<upcxx::rank_n(); j++) {
			cdouble* start = m_local + k*m_dims[0]*m_dims[1] + j*m_dims[0]*Ny;  
			INT loc = upcxx::rank_me()*Ny*m_dims[0]*m_Nz + k*m_dims[0]*Ny; 
			if (j == upcxx::rank_me()) {
				memcpy(tlocal+loc, start, m_dims[0]*m_Ny*sizeof(cdouble)); 
			} else {
				fut = upcxx::when_all(fut, 
					upcxx::rput(start, tmp[j]+loc, m_dims[0]*Ny)); 
			}
		}
	}
	fut.wait(); 
	CH_STOP(row); 
#elif defined PENCILS
	CH_TIMER("rows (OMP pencils)", row); 
	CH_START(row); 
	#pragma omp parallel
	{
		// threading workaround 
		upcxx::default_persona_scope();

		upcxx::future<> fut = upcxx::make_future();
		#pragma omp for  
		for (INT k=0; k<m_Nz; k++) {
			for (INT j=0; j<m_dims[1]; j++) {
				cdouble* row = m_local + j*m_dims[0] + k*m_dims[0]*m_dims[1]; 
				m_fft_x.transform(row, DIR); 
				INT send_to = j/Ny; 
				INT row_num = j % Ny;
				INT loc = upcxx::rank_me()*m_Nz*m_dims[0]*Ny + 
					k*m_dims[0]*Ny + 
					m_dims[0]*row_num; 
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
	CH_STOP(row); 
#else
	CH_TIMER("rows (serial pencils)", row); 
	CH_START(row); 
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
			if (send_to == upcxx::rank_me()) {
				memcpy(tlocal+loc, row, m_dims[0]*sizeof(cdouble)); 
			} else {
				fut = upcxx::when_all(fut, 
					upcxx::rput(row, tmp[send_to]+loc, m_dims[0])); 
			}
		}
	}
	fut.wait(); 
	CH_STOP(row); 
#endif

	upcxx::barrier(); 

	// --- transform in z direction --- 
#ifdef TRANSPOSE 
	transposeX2Z(tlocal); 
#endif
#ifdef OMP 
	CH_TIMER("z (OMP)", z); 
	CH_START(z); 
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
	CH_STOP(z); 
#else 
	CH_TIMER("z (serial)", z); 
	CH_START(z); 
	for (INT j=0; j<Ny; j++) {
		for (INT i=0; i<m_dims[0]; i++) {
			cdouble* zrow = tlocal + i + j*m_dims[0]; 
			m_fft_z.transform(zrow, DIR); 
		}
	}
	CH_STOP(z); 
#endif
#ifdef TRANSPOSE
	transposeZ2X(tlocal); 
#endif

	upcxx::barrier(); 

	// --- transpose back --- 
#ifdef OMP
	CH_TIMER("transpose back (OMP)", trans); 
	CH_START(trans); 
	#pragma omp parallel 
	{
		upcxx::default_persona_scope();

		upcxx::future<> f = upcxx::make_future(); 
		#pragma omp for 
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
	}
	CH_STOP(trans); 
#else 
	CH_TIMER("transpose back (serial)", trans); 
	CH_START(trans); 
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
	CH_STOP(trans); 
#endif

	CH_TIMER("clean up", clean); 
	CH_START(clean); 
	upcxx::delete_array(tmp[upcxx::rank_me()]);
	upcxx::barrier();
	CH_STOP(clean);  

	// for (int k=0; k<m_dims[2]; k++) {
	// 	for (int j=0; j<m_dims[1]; j++) {
	// 		m_fft_x.transform(m_local+m_dims[0]*j+m_dims[0]*m_dims[1]*k, DIR); 
	// 	}
	// }

	// for (int k=0; k<m_dims[2]; k++) {
	// 	for (int i=0; i<m_dims[0]; i++) {
	// 		m_fft_y.transform(m_local+i+k*m_dims[0]*m_dims[1], DIR); 
	// 	}
	// }

	// for (int j=0; j<m_dims[1]; j++) {
	// 	for (int i=0; i<m_dims[0]; i++) {
	// 		m_fft_z.transform(m_local+i+j*m_dims[1], DIR); 
	// 	}
	// }

}

void Scalar::transposeX2Z(cdouble* f) {
	CH_TIMERS("transpose to z contiguous"); 
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
	CH_TIMERS("transpose to x contiguous"); 
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

void Scalar::getIndex(array<INT,DIM> index, INT& rank, INT& loc) const {
	INT n = index[0] + m_dims[0]*index[1] + index[2]*m_dims[0]*m_dims[1]; 
	if (n >= m_N) {
		cout << n << endl; 
		ERROR("indexing out of full dimensions"); 
	}
	rank = n/m_dSize; // which rank owns the data 
	loc = n % m_dSize; // remainder is index into the local array 
}

int Scalar::m_nscalars=0; 