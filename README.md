# spectral_upcxx
Massively parallel spectral Navier Stokes solver (UPC++, OpenMP) 

Build by copying make.inc.in to make.inc and editing to match system. Must define where FFTW is and what compiler to use. 

# docs/
Reference papers and a doxygen setup. Build doxygen files with make docs in root directory. 

# exec/
## ns.cpp
Main Navier Stokes code. Uses a Fraction Step Method to evolve NS equations (AB2 on nonlinear and pressure terms, CN on viscous term). 
Initial conditions are vortices in a triply-periodic box. 

## fft.cpp
Times the 3D FFT on random data and checks for correctness. 

# test/ 
Various testing programs. Main test program is operators.cpp which checks curl, laplacian, gradient, etc for correctness. 

# src/ 
## FFT1D
Wrapper for FFTW that handles plan create, destruction, and execution. Stores plans for reuse. 

## DataObjects 
### Scalar 
Main distributed array class for storing a 3D scalar variable. Has 3D FFT functions as well as common operators 
such as gradient, Laplacian, and inverse Laplacian. 

### Vector 
Stores 3 Scalar objects to represent vector variables. Defines FFT, cross product, divergence, vector Laplacian, and curl. 

### Operators 
Operator overloaded arithmetic operators (+,-,*,etc.). 

## Writer 
Writes Scalars and Vectors to to VTK in parallel. Writes a Visit master file to tell Visit how to combine the parallel files. 
Stores a list of pointers to Scalar and Vectors and outputs all variables to one VTK file (for each UPC++ rank). 

## Particles 
Tracer particle class. Not yet parallelized. Use tri-linear interpolation to advect particles using the velocity field at each time step. 
Makes cool plots (Blender?). 

# utils/
Source code for dependencies. VisitWriter is the main VTK writer. chtimer contains CH_Timer a function tracer/timer (with edits for UPC++). 
