#include "CpuID.h" 
#include <mmintrin.h>		// MMX  
#include <xmmintrin.h>		// SSE(include mmintrin.h)  
#include <emmintrin.h>		// SSE2(include xmmintrin.h)  
#include <pmmintrin.h>		// SSE3(include emmintrin.h)  
#include <tmmintrin.h>		// SSSE3(include pmmintrin.h)  
#include <smmintrin.h>		// SSE4.1(include tmmintrin.h)  
#include <nmmintrin.h>		// SSE4.2(include smmintrin.h)  
#include <wmmintrin.h>		// AES(include nmmintrin.h)  
#include <immintrin.h>		// AVX(include wmmintrin.h)  
#include <intrin.h>			// (include immintrin.h)


void CpuID::Executecpuid(DWORD veax)
{ 
	// ��ΪǶ��ʽ�Ļ����벻��ʶ�����Ա����,���Զ����ĸ���ʱ������Ϊ���� 
	DWORD deax;
	DWORD debx;
	DWORD decx;
	DWORD dedx;
	__asm {
		mov eax, veax;		// �������������eax
		cpuid;				// ִ��cpuid
		mov deax, eax;		// �������д���ѼĴ����еı���������ʱ����
		mov debx, ebx
		mov decx, ecx
		mov dedx, edx
	}
	m_eax = deax;			// ����ʱ�����е����ݷ������Ա���� 
	m_ebx = debx;
	m_ecx = decx;
	m_edx = dedx;
}

string CpuID::GetVID()
{
	char cVID[13]; // �ַ����������洢��������Ϣ
	memset(cVID, 0, 13); // ��������0 
	Executecpuid(0); // ִ��cpuidָ�ʹ��������� eax = 0 
	memcpy(cVID, &m_ebx, 4); // ����ǰ�ĸ��ַ������� 
	memcpy(cVID + 4, &m_edx, 4); // �����м��ĸ��ַ������� 
	memcpy(cVID + 8, &m_ecx, 4); // ��������ĸ��ַ������� 
	return string(cVID); // ��string����ʽ���� 
}

string CpuID::GetBrand()
{
	const DWORD BRANDID = 0x80000002; // ��0x80000002��ʼ����0x80000004���� 
	char cBrand[49]; // �����洢�̱��ַ�����48���ַ� 
	memset(cBrand, 0, 49); // ��ʼ��Ϊ0 
	for (DWORD i = 0; i < 3; i++) // ����ִ��3��ָ�� 
	{
		Executecpuid(BRANDID + i);
		memcpy(cBrand + i * 16, &m_eax, 16); // ÿ��ִ�н����󣬱����ĸ��Ĵ������asc�뵽���� 
	} // �������ڴ��У�m_eax, m_ebx, m_ecx, m_edx���������� // ���Կ���ֱ�����ڴ�copy�ķ�ʽ���б��� 
	return string(cBrand); // ��string����ʽ���� 
}

bool CpuID::IsHyperThreading() // �ж��Ƿ�֧��hyper-threading 
{
	Executecpuid(1); // ִ��cpuidָ�ʹ��������� eax = 1 
	bool res = m_edx & (1 << 28);
	if (res) 
		printf("support HyperThreading\n");
	else 
		printf("don't support HyperThreading\n");
	return res; // ����edx��bit 28 
}

bool CpuID::IsEST() // �ж��Ƿ�֧��speed step 
{
	Executecpuid(1); // ִ��cpuidָ�ʹ��������� eax = 1 
	return m_ecx & (1 << 7); // ����ecx��bit 7 
} 

bool CpuID::IsMMX()
{
	Executecpuid(1); // ִ��cpuidָ�ʹ��������� eax = 1
	bool res = m_edx & (1 << 23);
	if (res) 
		printf("support MMX/n");
	else 
		printf("don't support MMX/n");
	return res; // ����edx��bit 23 
} 

bool CpuID::IsSSE() // �ж��Ƿ�֧��SSE 
{
	Executecpuid(1); // ִ��cpuidָ�ʹ��������� eax = 1
	bool res = m_edx & (1 << 25);
	if (res) 
		printf("support SSE\n");
	else 
		printf("don't support SSE\n");
	return res;
}

bool CpuID::IsSSE2() // �ж��Ƿ�֧��SSE2 
{
	Executecpuid(1); // ִ��cpuidָ�ʹ��������� eax = 1 
	bool res = m_edx & (1 << 26);
	if (res) 
		printf("support SSE2\n");
	else 
		printf("don't support SSE2\n");
	return res;
}

bool CpuID::IsSSE3() // �ж��Ƿ�֧��SSE3 
{
	Executecpuid(1); // ִ��cpuidָ�ʹ��������� eax = 1
	bool res = m_ecx;
	if (res) 
		printf("support SSE3\n");
	else 
		printf("don't support SSE3\n");
	return res;
}

bool CpuID::IsSSE4_1() // �ж��Ƿ�֧��SSE4.1 
{
	Executecpuid(1); // ִ��cpuidָ�ʹ��������� eax = 1
	bool res = m_ecx & (1 << 19);
	if (res) 
		printf("support SSE4.1\n");
	else 
		printf("don't support SSE4.1\n");
	return res;
}

bool CpuID::IsSSE4_2() // �ж��Ƿ�֧��SSE4.2 
{
	Executecpuid(1); // ִ��cpuidָ�ʹ��������� eax = 1 
	bool res = m_ecx & (1<<20); 
	if (res) 
		printf("support SSE4.2\n");
	else 
		printf("don't support SSE4.2\n");
	return res;
}