#include "Timer.H"
#include <iostream>

Timer::Timer(string name) {
	m_stop = false; 
	m_name = name; 
	m_ptr = nullptr; 
	if (upcxx::rank_me() == 0) {
		m_ptr = upcxx::new_array<chrono::duration<double>>(upcxx::rank_n()); 
	}
	m_ptr = upcxx::broadcast(m_ptr, 0).wait(); 
	m_start = chrono::system_clock::now(); 
}

Timer::~Timer() {
	if (!m_stop) stop(); 
}

void Timer::stop() {
	m_el = chrono::system_clock::now() - m_start; 
	upcxx::rput(m_el, m_ptr+upcxx::rank_me()).wait(); 
	upcxx::barrier(); 
	double max = 0; 
	double min = 100000; 
	double average = 0; 
	if (upcxx::rank_me() == 0) {
		chrono::duration<double>* local = m_ptr.local(); 
		for (int i=0; i<upcxx::rank_n(); i++) {
			if (local[i].count() > max) max = local[i].count(); 
			if (local[i].count() < min) min = local[i].count(); 
			average += local[i].count(); 
		}
		average /= upcxx::rank_n(); 
		cout << m_name << endl << 
			"\tAverage: " << average << " seconds" << endl << 
			"\tMax: " << max << endl << 
			"\tMin: " << min << endl; 
	}

	m_stop = true; 

	// TimerData data; 
	// data.name = m_name; 
	// data.average = m_el.count(); 

	// TIMER_PARENT.add(data); 
}