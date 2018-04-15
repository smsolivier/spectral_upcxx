#ifndef _TIMER_PARENT_H_
#define	_TIMER_PARENT_H_

#include "TimerData.H"
#include <string> 
#include <vector> 
#include <chrono> 

using namespace std; 

class TimerParent {
public:
	static TimerParent& instance() {
		static TimerParent tp; 
		return tp; 
	}
	void add(TimerData data); 
	void print(); 
	void start(); 
private:
	TimerParent(); 
	~TimerParent(); 
	vector<TimerData> m_timers; 
	chrono::time_point<chrono::system_clock> m_start; 
	chrono::duration<double> m_el; 
	bool m_wtime; 
};  

#endif