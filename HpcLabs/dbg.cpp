#include <windows.h>
#include <crtdbg.h>
#include <DbgHelp.h>
#pragma comment(lib,  "dbghelp.lib")
#include "dbg.h"

void DetectMemoryLeaks(bool on_off)
{
	int flags = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	if (!on_off)
		flags &= ~_CRTDBG_LEAK_CHECK_DF;
	else
	{
		flags |= _CRTDBG_LEAK_CHECK_DF;
		_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
		_CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
	}
	_CrtSetDbgFlag(flags);
}


CCreateDump* CCreateDump::m_instance = NULL;
std::string CCreateDump::m_dumpFile = "";

long CCreateDump::UnhandleExceptionFilter(_EXCEPTION_POINTERS* ExceptionInfo)
{
	HANDLE hFile = CreateFileA(m_dumpFile.c_str(), GENERIC_WRITE, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile != INVALID_HANDLE_VALUE)
	{
		MINIDUMP_EXCEPTION_INFORMATION   ExInfo;
		ExInfo.ThreadId = ::GetCurrentThreadId();
		ExInfo.ExceptionPointers = ExceptionInfo;
		ExInfo.ClientPointers = FALSE;
		//   write   the   dump
		BOOL   bOK = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal, &ExInfo, NULL, NULL);
		CloseHandle(hFile);
		if (!bOK)
		{
			DWORD dw = GetLastError();
			//写dump文件出错处理,异常交给windows处理
			return EXCEPTION_CONTINUE_SEARCH;
		}
		else
		{    //在异常处结束
			return EXCEPTION_EXECUTE_HANDLER;
		}
	}
	else
	{
		return EXCEPTION_CONTINUE_SEARCH;
	}
}

void CCreateDump::DeclarDumpFile(std::string dmpFileName)
{
	SYSTEMTIME syt;
	GetLocalTime(&syt);
	char c[MAX_PATH];
	sprintf_s(c, MAX_PATH, "[%04d-%02d-%02d %02d：%02d：%02d]", syt.wYear, syt.wMonth, syt.wDay, syt.wHour, syt.wMinute, syt.wSecond);
	m_dumpFile = std::string(c);
	if (!dmpFileName.empty())
	{
		m_dumpFile += dmpFileName;
	}
	m_dumpFile += std::string(".dmp");
	SetUnhandledExceptionFilter(UnhandleExceptionFilter);
}

CCreateDump* CCreateDump::Instance()
{
	if (m_instance == nullptr)
	{
		m_instance = new CCreateDump;
	}
	return m_instance;
}