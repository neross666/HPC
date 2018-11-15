#include "Profiler.h"
#include "function.h"
#include <assert.h>

namespace TimerSys
{
#ifdef _GH_Gh
	void _cdecl enterFunc(long retAddress)
	{
		Profiler::Instance()->enterFunc(retAddress);
	}

	void _cdecl exitFunc(long retAddress)
	{
		Profiler::Instance()->exitFunc(retAddress);
	}

	extern "C" void __declspec(naked) _cdecl _penter(void)
	{
		_asm
		{
			//Prolog instructions
			pushad
			//calculate the pointer to the return address by adding 4*8 bytes 
			//(8 register values are pushed onto stack which must be removed)
			mov  eax, esp
			add  eax, 32
			// retrieve return address from stack
			mov  eax, dword ptr[eax]
			// subtract 5 bytes as instruction for call _penter is 5 bytes long on 32-bit machines, e.g. E8 <00 00 00 00>
			sub  eax, 5
			// provide return address to recordFunctionCall
			push eax
			call enterFunc
			pop eax

			//Epilog instructions
			popad
			ret
		}
	}

	extern "C" void __declspec(naked) _cdecl _pexit(void)
	{
		_asm
		{
			//Prolog instructions
			pushad

			//calculate the pointer to the return address by adding 4*7 bytes 
			//(7 register values are pushed onto stack which must be removed)
			mov  eax, esp
			add  eax, 32

			// retrieve return address from stack
			mov  eax, dword ptr[eax]

			// subtract 5 bytes as instruction for call _penter is 5 bytes long on 32-bit machines, e.g. E8 <00 00 00 00>
			sub  eax, 5

			// provide return address to recordFunctionCall
			push eax
			call exitFunc
			pop eax

			//Epilog instructions
			popad
			ret
		}
	}
#endif

	//---------------------------------------------------
	Profiler* Profiler::Instance()
	{
		static Profiler s_instanse;
		return &s_instanse;	
	}

	Profiler::Profiler(void) : 
		m_Function(), 
		m_callbackFunc(nullptr), 
		m_enable(true)
	{
		m_mutex = new std::mutex();
	}

	Function* Profiler::getFunc(std::string name)
	{
		std::map<std::string, Function*>::iterator iter = m_Function.find(name);
		if (iter != m_Function.end())
		{
			return iter->second;
		}
		else
		{
			Function* pFunc = new Function(name);
			m_Function.insert( std::make_pair(name, pFunc));
			return pFunc;
		}
	}

	Profiler::~Profiler(void)
	{
		clear();

		if(m_mutex != nullptr)
		{
			delete m_mutex;
			m_mutex = nullptr;
		}
	}

	void Profiler::enable(bool enble)
	{
		m_mutex->try_lock();
		if (enble != m_enable)
		{
			m_enable = enble;
			if (!enble)
			{
				clear();
			}
		}
		m_mutex->unlock();
	}

	void Profiler::enterFunc(std::string name)
	{
		m_mutex->try_lock();

		if (!m_enable)
			return;

		Function *func = getFunc(name);
		func->startTimer();
		m_callStack.push(func);
	}

	void Profiler::exitFunc(std::string name)
	{
		if (!m_enable)
		{
			m_mutex->unlock();
			return;
		}

		Function* func = m_callStack.top();
		assert(func->getName() == name);

		float time = 0.0;
		func->stopTimer(time);
		func->addTime(time);
		
		m_callStack.pop();
		if (!m_callStack.empty())
		{
			Function* func_b = m_callStack.top();
			func->m_parent = func_b;
			func_b->m_chilren.insert(func);
		}

		if (m_callbackFunc != NULL)
		{
			m_callbackFunc(name, time);
		}

		m_mutex->unlock();
	}

	void Profiler::getProfiler(std::map<std::string, Function*>& function)
	{
		function = m_Function;
	}

	void Profiler::clear()
	{
		std::map<std::string, Function*>::iterator Iter = m_Function.begin();
		for ( ;Iter != m_Function.end(); Iter++)
		{
			delete Iter->second;
		}
		m_Function.clear();
	}

	void Profiler::constructCallTree()
	{

	}

}

