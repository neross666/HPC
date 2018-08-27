#pragma once
#include <string>
#include <map>


using namespace std;


class TimeClock
{
public:
	enum Unit
	{
		Minute = 1,
		Second,
		Millisecond,
		Microsecond,
	};

public:
	TimeClock(string algName, Unit unit=Millisecond);
	~TimeClock();

	static void RemoveAlg(string algName);

private:
	void ShowTimeClock();

private:
	Unit m_uint;

	long long m_st;
	long long m_ed;
	long long m_freq;

	string m_curAlg;

	static map<string, double> m_avgTime;
	static map<string, int> m_count;
	static map<string, double> m_totalTime;
};