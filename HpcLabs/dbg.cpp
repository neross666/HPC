#include <windows.h>
#include <crtdbg.h>
#include <DbgHelp.h>
#include <vector>
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

void PrintMemDistribute()
{
	typedef struct MemNoed
	{
		int node_saddr;
		int node_size;
		int node_type;
	};
	typedef struct MemInfo
	{
		DWORD dwTotalFreeSize;
		DWORD dwTotalCommitSize;
		DWORD dwTotalReserveSize;

		vector<MemNoed> nodes;

		void print()
		{
			HANDLE hdl = GetStdHandle(STD_OUTPUT_HANDLE);
			printf("Total Free Memory Size:%f MB\n", (float)dwTotalFreeSize / 1024 / 1024);
			printf("Total Commit Memory Size:%f MB\n", (float)dwTotalCommitSize / 1024 / 1024);
			printf("Total Reserve Memory Size:%f MB\n", (float)dwTotalReserveSize / 1024 / 1024);

			DWORD total = dwTotalFreeSize + dwTotalCommitSize + dwTotalReserveSize;
			string progressbar;
			size_t percent;
			for (size_t i = 0; i < nodes.size(); i++)
			{
				switch (nodes[i].node_type)
				{
				case MEM_COMMIT:
					SetConsoleTextAttribute(hdl, FOREGROUND_RED);
					break;
				case MEM_RESERVE:
					SetConsoleTextAttribute(hdl, FOREGROUND_GREEN);
					break;
				case MEM_FREE:
					SetConsoleTextAttribute(hdl, FOREGROUND_BLUE);
					break;
				}
				percent = (size_t)(2048.0f*nodes[i].node_size / total);
				for (size_t j = 0; j < percent; j++)
				{
					printf("#");
				}
				printf("%d ", nodes[i].node_size / 1024);
			}
			printf("\n");
			SetConsoleTextAttribute(hdl, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
		}
	};


	SIZE_T retVal; // VirtualQuery返回值
	MEMORY_BASIC_INFORMATION mbi; // 返回页面信息
	DWORD dwStartAddress = 0x0; // 起始地址
	DWORD dwTotalFreeSize = 0; // 空闲页面总大小
	DWORD dwTotalCommitSize = 0;
	DWORD dwTotalReserveSize = 0;

	MemInfo mem_info;

	while (true)
	{
		MemNoed node;

		ZeroMemory(&mbi, sizeof(MEMORY_BASIC_INFORMATION));
		retVal = VirtualQuery((LPCVOID)dwStartAddress, &mbi, sizeof(MEMORY_BASIC_INFORMATION));

		if (0 == retVal) // 返回0表示失败
			break;

		// 判断mbi中的State标识，累加为MEM_FREE的区间
		if (MEM_FREE == mbi.State)
		{
			dwTotalFreeSize += mbi.RegionSize;
		}
		else if (MEM_COMMIT == mbi.State)
		{
			dwTotalCommitSize += mbi.RegionSize;
		}
		else if (MEM_RESERVE == mbi.State)
		{
			dwTotalReserveSize += mbi.RegionSize;
		}
		node.node_saddr = dwStartAddress;
		node.node_size = mbi.RegionSize;
		node.node_type = mbi.State;
		mem_info.nodes.push_back(node);

		// 下一个区间
		dwStartAddress += mbi.RegionSize;
	}
	mem_info.dwTotalCommitSize = dwTotalCommitSize;
	mem_info.dwTotalReserveSize = dwTotalReserveSize;
	mem_info.dwTotalFreeSize = dwTotalFreeSize;
	mem_info.print();
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
		// write the dump
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