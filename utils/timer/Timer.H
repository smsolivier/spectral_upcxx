#ifndef _TIMER_H_
#define _TIMER_H_

#include "TimerData.H"
#include "TimerParent.H"
#include <chrono> 
#include <string> 

using namespace std; 

class Timer {
public:
	Timer(string name); 
	~Timer(); 
	void stop(); 
private:
	chrono::time_point<chrono::system_clock> m_start; 
	chrono::duration<double> m_el; 
	TimerData m_data; 
	bool m_alive; 
	bool m_go; 
	bool m_verbose; 
	string m_name; 
}; 

#define START_WTIMER() TimerParent::instance().start(); 

#endif