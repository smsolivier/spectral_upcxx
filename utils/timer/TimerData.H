#ifndef _TIMER_DATA_H_
#define _TIMER_DATA_H_

#include <string> 
#include <fstream> 

using namespace std; 

class TimerData {
public:
	TimerData(); 
	void print(ofstream& out); 
	void update(TimerData data); 

	string name; 
	double elapsed; 
	int count; 
}; 

#endif