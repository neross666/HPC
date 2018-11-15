VS2010 32bit

编译器配置：
a、选择需要打印日志的源文件
b、右键属性
c、C/C++ ---> 命令行 ---> 其他选项中输入 /Gh /GH。注，VS编译器中若使用了 /clr 命令，则无法再使用 /Gh 和 /GH
	/Gh：在每个函数的开始处添加_penter函数
	/GH：在每个函数的开始处添加_pexit函数
d、打开项目属性 ---> 连接器 ---> 常规 ---> 启用增量链接:否
	该步骤用于避免函数指针所指地址与函数第一条指令地址不一致

_penter函数原型：
extern "C" void __declspec(naked) _cdecl _penter( void ) 

_pexit函数原型：
extern "C" void __declspec(naked) _cdecl _pexit( void ) 

_penter所在源文件不能使用/Gh选项
_pexit所在源文件不能使用/GH选项


在_penter函数中获取调用函数的起始地址：
extern "C" void __declspec(naked) _cdecl _penter( void ) 
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

在_pexit函数中通过汇编只能获取调用函数结束地址：
extern "C" void __declspec(naked) _cdecl _pexit( void ) 
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


构建函数调用栈



设置过滤器


