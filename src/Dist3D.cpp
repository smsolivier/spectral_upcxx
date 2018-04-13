#include "Dist3D.H" 
#include <iostream>
#ifdef OMP 
#include <omp.h> 
#endif
#include "Timer.H"

Dist3D::Dist3D() {}
Dist3D::~Dist3D() {
	upcxx::delete_array(m_ptrs[upcxx::rank_me()]); 
}

Dist3D::Dist3D(array<INT,DIM> N) {
	init(N); 
}

void Dist3D::init(array<INT,DIM> N) {
	m_dims = N; 

	// compute total size 
	m_N = 1; 
	for (INT i=0; i<DIM; i++) {
		m_N *= m_dims[i]; 
	}

	// number of x,y slabs per rank 
	m_Nz = ceil((double)m_dims[2]/upcxx::rank_n()); 

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
}

void Dist3D::set(array<INT,DIM> index, cdouble val) {
	INT rank, loc; 
	getIndex(index, rank, loc); 

	// only rput if not local data 
	if (rank == upcxx::rank_me()) {
		m_local[loc] = val; 
	} else {
		upcxx::rput(val, m_ptrs[rank]+loc).wait(); 
	}
}

cdouble Dist3D::operator[](array<INT,DIM> index) {
	INT rank, loc; 
	getIndex(index, rank, loc); 

	// only do rget if not a local index 
	if (rank == upcxx::rank_me()) {
		return m_local[loc]; 
	} else {
		return upcxx::rget(m_ptrs[rank]+loc).wait(); 
	}
}

cdouble& Dist3D::operator[](INT index) {
	if (index >= m_dSize) {
		cout << "ERROR (Dist3D.cpp): index out of range in direct access operator" << endl; 
	}
	return m_local[index]; 
}

void Dist3D::forward() {
	transform(1); 
}

void Dist3D::inverse() {
	transform(-1); 

	// divide by N^3 
	// #pragma omp parallel for 
	for (INT i=0; i<m_dSize; i++) {
		m_local[i] /= m_N; 
	}
}

void Dist3D::add(Dist3D& a) {
	#pragma omp parallel for schedule(static)
	for (INT i=0; i<m_dSize; i++) {
		m_local[i] += a[i];  
	}
}

INT Dist3D::sizePerProcessor() {return m_Nz; } 
INT Dist3D::size() {return m_N; } 
cdouble* Dist3D::getLocal() {return m_local; } 
double Dist3D::memory() {
	if (upcxx::rank_me() == 0) {
		cout << "memory requirement = " << (double)m_N*sizeof(cdouble)/1e9 << " GB" << endl; 
	}
}

void Dist3D::transform(INT DIR) {
	// store transposed data in tmp 
	vector<upcxx::global_ptr<cdouble>> tmp(upcxx::rank_n(), NULL);
	// allocate global memory 
	tmp[upcxx::rank_me()] = upcxx::new_array<cdouble>(m_dSize);
	// broadcast to other ranks 
	for (INT i=0; i<upcxx::rank_n(); i++) {
		tmp[i] = upcxx::broadcast(tmp[i], i).wait(); 
	}
	// get local access 
	cdouble* tlocal = tmp[upcxx::rank_me()].local(); 

	// --- transform columns --- 
#ifdef OMP 
	#pragma omp parallel 
	{
		FFT1D fft(m_dims[1], m_dims[0], DIR); 
		#pragma omp master 
		{
			cout << "nthreads = " << omp_get_num_threads() << endl; 
		}
		#pragma omp for 
		for (INT k=0; k<m_Nz; k++) {
			for (INT i=0; i<m_dims[0]; i++) {
				cdouble* start = m_local + i + k*m_dims[0]*m_dims[1]; 
				// fft.transform(start); 
				cout << "transform " << i << ", " << k << endl; 
				fft.transform(start, m_dims[1], m_dims[0], DIR);
			}
		}
	}
#else
	for (INT k=0; k<m_Nz; k++) {
		for (INT i=0; i<m_dims[0]; i++) {
			m_fft.transform(m_local+i+k*m_dims[0]*m_dims[1], 
				m_dims[1], m_dims[0], DIR); 
		}
	}
#endif
	upcxx::barrier(); 

	// --- transforms rows and pencil transpose --- 
	INT Ny = m_dims[1]/upcxx::rank_n(); 
#ifdef OMP 
	#pragma omp parallel 
	{
		FFT1D fft(m_dims[0], 1, DIR); 
		#pragma omp master
		{
			cout << "starting columns" << endl; 
		}
		#pragma omp for 
		for (INT k=0; k<m_Nz; k++) {
			for (INT j=0; j<m_dims[1]; j++) {
				cdouble* start = m_local + j*m_dims[0] + k*m_dims[0]*m_dims[1]; 
				fft.transform(start, m_dims[0], 1, DIR); 

				// pencil transpose 
				INT send_to = j/Ny; 
				INT row_num = j % Ny;
				INT loc = upcxx::rank_me()*m_Nz*m_dims[0]*Ny + 
					k*m_dims[0]*Ny + 
					m_dims[0]*row_num; 
				if (loc >= m_N) cout << "out of range" << endl; 
				if (send_to == upcxx::rank_me()) {
					memcpy(tlocal+loc, start, m_dims[0]*sizeof(cdouble)); 
				} else {
					upcxx::rput(start, tmp[send_to]+loc, m_dims[0]); 				
				}
			}
		}
	}
#else
	for (INT k=0; k<m_Nz; k++) {
		for (INT j=0; j<m_dims[1]; j++) {
			cdouble* row = m_local + j*m_dims[0] + k*m_dims[0]*m_dims[1]; 
			m_fft.transform(row, m_dims[0], 1, DIR); 
			INT send_to = j/Ny; 
			INT row_num = j % Ny;
			INT loc = upcxx::rank_me()*m_Nz*m_dims[0]*Ny + 
				k*m_dims[0]*Ny + 
				m_dims[0]*row_num; 
			if (loc >= m_N) cout << "out of range" << endl; 
			if (send_to == upcxx::rank_me()) {
				memcpy(tlocal+loc, row, m_dims[0]*sizeof(cdouble)); 
			} else {
				upcxx::rput(row, tmp[send_to]+loc, m_dims[0]); 				
			}
		}
	}
#endif
	upcxx::barrier(); 

	// --- transform in z direction --- 
#ifdef OMP 
	#pragma omp parallel 
	{
		FFT1D fft(m_dims[2], m_dims[0]*Ny, DIR); 
		#pragma omp master 
		{
			cout << "starting z" << endl; 
		}
		#pragma omp for 
		for (INT j=0; j<Ny; j++) {
			for (INT i=0; i<m_dims[0]; i++) {
				cdouble* start = tlocal + i + j*m_dims[0]; 
				fft.transform(start, m_dims[2], m_dims[0]*Ny, DIR); 
			}
		}
	}
#else 
	for (INT j=0; j<Ny; j++) {
		for (INT i=0; i<m_dims[0]; i++) {
			cdouble* zrow = tlocal + i + j*m_dims[0]; 
			m_fft.transform(zrow, m_dims[2], m_dims[0]*Ny, DIR); 
		}
	}
#endif
	upcxx::barrier(); 

	// switch back to original data layout (send slabs of Nx*Ny) 
	for (INT k=0; k<m_dims[2]; k++) {
		cdouble* slab = tlocal + k*Ny*m_dims[0]; 
		INT send_to = k/m_Nz; 
		INT rem = k % m_Nz; 
		INT dest = upcxx::rank_me()*Ny*m_dims[0] + 
			rem*m_dims[0]*m_dims[1]; 
		if (send_to == upcxx::rank_me()) {
			memcpy(m_local+dest, slab, m_dims[0]*Ny*sizeof(cdouble)); 
		} else {
			upcxx::rput(slab, m_ptrs[send_to]+dest, m_dims[0]*Ny).wait();
		}
	}

	// clean up temp variable 
	upcxx::delete_array(tmp[upcxx::rank_me()]);
	upcxx::barrier(); 
}

void Dist3D::transpose() {
	cdouble* tmp = new cdouble[m_N];
	memcpy(tmp, m_local, m_N*sizeof(cdouble)); 
	array<INT,DIM> ind, indt = {0,0,0}; 
	for (INT i=0; i<m_dims[0]; i++) {
		for (INT j=0; j<m_dims[1]; j++) {
			for (INT k=0; k<m_Nz; k++) {
				m_local[k+m_Nz*j+m_Nz*m_dims[1]*i] = 
					tmp[i+j*m_dims[0]+k*m_dims[0]*m_dims[1]]; 
			}
		}
	}
	delete(tmp); 
}

void Dist3D::getIndex(array<INT,DIM> index, INT& rank, INT& loc) {
	INT n = index[0] + m_dims[0]*index[1] + index[2]*m_dims[0]*m_dims[1]; 
	if (n >= m_N) {
		cout << "ERROR: index out of range in Dist3D.cpp" << endl; 
		upcxx::finalize(); // close upcxx 
		exit(0); // exit program 
	}
	rank = n/m_dSize; // which rank owns the data 
	loc = n % m_dSize; // remainder is index into the local array 
}