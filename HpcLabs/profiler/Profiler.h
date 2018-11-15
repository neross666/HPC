#pragma once
#include <mutex>
#include <map>
#include <stack>

//#define _GH_Gh

#define TIMING TimerSys::Timing(__FUNCTION__);


typedef void (*pCallBack)(std::string name, float cost);

namespace TimerSys
{
	class Function;

	class Profiler
	{
	public:
		static Profiler *Instance();
		~Profiler();


		void enable(bool enble);

		void enterFunc(long va);
		void enterFunc(std::string name );

		void exitFunc(long va);
		void exitFunc(std::string name );

		void getProfiler(std::map<std::string, Function*>& function);

		void clear();

		void constructCallTree();

	private:
		Profiler();

		Function* getFunc( std::string name );
		
	private:
		std::mutex*							m_mutex;
		
		std::map<std::string, Function*>	m_Function;

		pCallBack							m_callbackFunc;

		std::stack<Function*>				m_callStack;

		bool								m_enable;
	};

	class Timing
	{
	public:
		Timing(std::string name) : m_name(name)
		{
			Profiler::Instance()->enterFunc(name);
		}
		~Timing()
		{
			Profiler::Instance()->exitFunc(m_name);
		}

	private:
		std::string m_name;
	};
}


