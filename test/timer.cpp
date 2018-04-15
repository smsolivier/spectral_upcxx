#include "Timer.H"
#include <iostream>
#include <unistd.h>
#include <upcxx/upcxx.hpp> 

using namespace std; 

int main() {
	upcxx::init(); 
	START_WTIMER();   
	cout << "before test" << endl; 
	{
		Timer timer("test"); 
	}

	upcxx::finalize(); 
}