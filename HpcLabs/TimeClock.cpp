#include "TimeClock.h"
#include <iostream>
#include <windows.h>

std::map<std::string, double> TimeClock::m_avgTime;
std::map<std::string, int> TimeClock::m_count;
std::map<std::string, double> TimeClock::m_totalTime;



TimeClock::TimeClock(string algName, Unit unit/*=Millisecond*/)
{
	m_curAlg = algName;
	m_uint = unit;
	if (m_count.find(algName) == m_count.end())
	{
		m_avgTime[algName] = 0.0f;
		m_count[algName] = 0;
		m_totalTime[algName] = 0;
	}

	m_st = 0;
	QueryPerformanceCounter((LARGE_INTEGER *)&m_st);
	QueryPerformanceFrequency((LARGE_INTEGER *)&m_freq);
}


TimeClock::~TimeClock()
{
	m_ed = 0;
	QueryPerformanceCounter((LARGE_INTEGER *)&m_ed);
	long long diff = m_ed - m_st;
	double cur_time_s = diff / (double)m_freq;

	m_count[m_curAlg] += 1;
	m_totalTime[m_curAlg] += cur_time_s;
	m_avgTime[m_curAlg] = m_totalTime[m_curAlg] / m_count[m_curAlg];// unit:s

	ShowTimeClock();
}


void TimeClock::RemoveAlg(string algName)
{
	auto it_c = m_count.find(algName);
	if (it_c != m_count.end())
		m_count.erase(it_c);

	auto it_a = m_totalTime.find(algName);
	if (it_a != m_totalTime.end())
		m_totalTime.erase(it_a);

	auto it_t = m_avgTime.find(algName);
	if (it_t == m_avgTime.end())
		m_avgTime.erase(it_t);
}


void TimeClock::ShowTimeClock()
{
	cout << "+++++++++++++" << m_curAlg << "+++++++++++++++++\n";
	switch (m_uint)
	{
	case Minute:
		cout << "average time :" << (long)(m_avgTime[m_curAlg]/60) << "min" << endl;
		break;
	case Second:
		cout << "average time :" << (long)(m_avgTime[m_curAlg]) << "s" << endl;
		break;
	case Millisecond:
		cout << "average time :" << (long long)(1000*m_avgTime[m_curAlg]) << "ms" << endl;
		break;
	case Microsecond:
		cout << "average time :" << (long long)(1000*1000 * m_avgTime[m_curAlg]) << "us" << endl;
		break;
	}	
	cout << "\n";
}

