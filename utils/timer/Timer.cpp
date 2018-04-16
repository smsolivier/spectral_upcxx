#include "Timer.H"
#include "TimerParent.H"
#include <fstream> 
#include <iostream> 
#include <upcxx/upcxx.hpp> 

/* --- Timer definitions --- */ 
Timer::Timer(string name) {
	m_alive = true; 
	m_go = false; 
	m_verbose = false; 
	m_name = name; 
	if (getenv("TIMER") != NULL) {
		m_go = true; 

		m_data.name = name; 
		m_start = chrono::system_clock::now(); 		
	}

	if (getenv("TIMER_VERBOSE") != NULL) {
		m_verbose = true; 
	}
}

Timer::~Timer() {
	if (m_alive) stop(); 
}

void Timer::stop() {
	if (!m_go) return; 

	m_el = chrono::system_clock::now() - m_start; 

	m_data.elapsed = m_el.count(); 
	TimerParent::instance().add(m_data); 

	if (m_verbose) {
		if (upcxx::rank_me() == 0) {
			cout << m_name << ": " << m_el.count() << endl; 			
		}
	}
	m_alive = false; 
}
/* --- end Timer --- */ 


/* --- TimerParent definitions --- */ 
TimerParent::TimerParent() {
}

void TimerParent::start() {
	m_start = chrono::system_clock::now(); 
	m_wtime = true; 
}

TimerParent::~TimerParent() {
	if (m_wtime && upcxx::rank_me() == 0) {
		m_el = chrono::system_clock::now() - m_start; 
		cout << "Wall Time = " << m_el.count() << " seconds" << endl; 		
	}

	if (getenv("TIMER") != NULL) 
		print(); 
}

void TimerParent::add(TimerData data) {
	int index = -1; 
	for (int i=0; i<m_timers.size(); i++) {
		if (data.name.compare(m_timers[i].name) == 0) {
			index = i; 
		}
	}

	if (index == -1) {
		m_timers.push_back(data); 
	} else {
		m_timers[index].update(data); 
	}
}

void TimerParent::print() {
	string fname = "r"+to_string(upcxx::rank_me())+".time"; 
	ofstream out(fname); 
	for (int i=0; i<m_timers.size(); i++) {
		m_timers[i].print(out); 
	}
}
/* --- end TimerParent --- */ 


/* --- TimerData definitions --- */ 
TimerData::TimerData() {
	count = 1; 
}

void TimerData::print(ofstream& out) {
	out << name << "(" << count << "): " << elapsed << " seconds" << endl;  
}

void TimerData::update(TimerData data) {
	count++; 
	elapsed += data.elapsed; 
}
/* --- end TimerData --- */ 