#pragma once
#include <string>

using namespace std;
#define _MEMCHKE

#ifdef _DEBUG

#ifdef _MEMCHKE
#define new new(_CLIENT_BLOCK, __FILE__, __LINE__)
#endif

void DetectMemoryLeaks(bool on_off);


class CCreateDump
{
public:	
	~CCreateDump(void) {}
	static CCreateDump* Instance();
	static long __stdcall UnhandleExceptionFilter(_EXCEPTION_POINTERS* ExceptionInfo);
	void DeclarDumpFile(std::string dmpFileName = "");

private:
	CCreateDump() {}

private:
	static std::string    m_dumpFile;
	static CCreateDump*   m_instance;
};

#endif // _DEBUG