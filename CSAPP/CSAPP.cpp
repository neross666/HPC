#include "pch.h"
#include <iostream>

// 判断两个有符号整型数据相加是否溢出
bool tadd_ok(int x, int y)
{
	int sum = x + y;
	bool ret = (x > 0) && (y > 0) && (sum <= 0);
	bool ret1 = (x < 0) && (y < 0) && (sum >= 0);
	return !(ret || ret1);
}

bool tadd_ok1(char x, char y)
{
	char sum = x + y;
	
	// bug:当sum-x超出char的表达范围时，被截断。无论x+y是否溢出，总是返回true
// 	char c = sum - x;
// 	char d = sum - y;
// 	return (c == y) && (d == x);


	return (sum - x == y) && (sum - y == x);	// 此时sum-x表达式的值为int型，因为sum和x或y是char型，不存在溢出截断的操作
}

bool tadd_ok2(int x, int y)
{
	int sum = x + y;
	return (sum - x == y) && (sum - y == x);// 此时sum-x表达式的值为int型，因为sum和x或y都是int型，因此存在溢出截断的操作

}

int main()
{
	bool ok = tadd_ok(0x80000000, 0x80000000);
	bool ok1 = tadd_ok1(64, 64);
	bool ok2 = tadd_ok2(0x80000000, 0x80000000);	
}
