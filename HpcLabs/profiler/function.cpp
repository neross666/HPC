#include "Function.h"

namespace TimerSys
{
	Function::Function( std::string name) 
		:	m_funcName( name), 
		m_totalTime(0.0), 
		m_numCalls(0), 
		m_numIgnore(1),
		m_startTime(0),
		m_endTime(0),
		m_curTime(0.0f),
		m_avgTime(0.0f),
		m_parent(NULL),
		m_chilren()
	{
	}
	
	void Function::addTime( float time)
	{
		m_curTime = time;
		m_numCalls++;

		if (m_numIgnore>0)
		{
			m_numIgnore--;
		}else
		{
			m_totalTime += time;
			m_avgTime = m_totalTime/(m_numCalls-m_numIgnore);
		}
	}

	float Function::getTotalTime() const
	{
		return m_totalTime;
	}

	float Function::getCurTime() const
	{
		return m_curTime;
	}

	float Function::getAvgTime() const
	{
		return m_avgTime;
	}

	std::string Function::getName() const
	{
		return m_funcName;
	}

	int	 Function::getCalls() const
	{
		return m_numCalls;
	}

	void Function::startTimer()
	{
		m_startTime = clock();
	}

	// Unit: ms
	void Function::stopTimer( float &time)
	{
		m_endTime = clock();
		time = (float) ( m_endTime - m_startTime) / CLOCKS_PER_SEC * 1000;
		m_startTime = 0;
	}	
}