#ifndef __TIMER_H__
#define __TIMER_H__
#include <chrono> 
#include <string> 
#include <upcxx/upcxx.hpp> 

using namespace std; 

class Timer {
public: 
	Timer(string name);
	void stop(); 
private:
	chrono::time_point<chrono::system_clock> m_start; 
	chrono::duration<double> m_el; 
	upcxx::global_ptr<chrono::duration<double>> m_ptr; 
	string m_name;  
}; 

#endif