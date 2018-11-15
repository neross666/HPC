#pragma once
#include <set>
#include <time.h>

namespace TimerSys
{
	class Function
	{
	public:	
		Function( std::string name);
		
		void addTime( float time);

		float getTotalTime() const;

		float getCurTime() const;

		float getAvgTime() const;
		
		std::string getName() const;

		int	 getCalls() const;

		void startTimer();

		void stopTimer( float& time);

		long long getStartTime(){return m_startTime;}

		long long getEndTime(){return m_endTime;}
						
	protected:
		std::string		m_funcName;
		int				m_numCalls;
		int				m_numIgnore;
		float			m_totalTime;
		float			m_curTime;
		float			m_avgTime;
		clock_t			m_startTime;
		clock_t			m_endTime;

	public:
		Function*		m_parent;
		std::set<Function*> m_chilren;
	};
}


