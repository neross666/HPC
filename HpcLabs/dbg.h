#pragma once
#include <string>

using namespace std;
#define _MEMCHKE

#ifdef _DEBUG

#ifdef _MEMCHKE
#define new new(_CLIENT_BLOCK, __FILE__, __LINE__)
#endif

void DetectMemoryLeaks(bool on_off);

void PrintMemDistribute();


// 生成的dmp文件与exe、pdb文件在放同一个目录，使用vs打开dmp文件，运行即可查看到程序最后发生错误的位置
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