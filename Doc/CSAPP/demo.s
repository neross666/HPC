1、包含参数构造区和局部变量的运行时栈结构：
-------------------------------------------------------------------
void proc(long a1, long *a1p, 
	int a2, int* a2p,
	short a3, short* a3p,
	char a4, char* a4p)
{
	*a1p += a1;
	*a2p += a2;
	*a3p += a3;
	*a4p += a4;
}
long multi()
{
	long a1  =12;
	int a2   =23;
	short a3 =34;
	char a4  =45;
	proc(a1, &a1, a2, &a2, a3, &a3, a4, &a4);
	return (a1+a2)*(a3-a4);
}
------------------------------------------------------------------
proc:	// 执行call proc之后，&a1相对%rsp偏移40；&a2相对%rsp偏移36；&a3相对%rsp偏移34；&a4相对%rsp偏移33
	movq	16(%rsp), %rax		// &a4 移到 %rax
	addq	%rdi, (%rsi)		// *a1p += a1;
	addl	%edx, (%rcx)		// *a2p += a2;
	addw	%r8w, (%r9)			// *a3p += a3;
	movl	8(%rsp), %edx		// a4 移到 %edx
	addb	%dl, (%rax)			// *a4p += a4;
	ret 						// 	弹出栈顶，即返回地址，并将其赋给PC，PC=(%rsp); %rsp += 8; 
	// 执行ret之后，&a1相对%rsp偏移32；&a2相对%rsp偏移28；&a3相对%rsp偏移26；&a4相对%rsp偏移25

multi:
	subq	$40, %rsp			// 分配空间，用于存储局部变量等
	movq	$12, 16(%rsp)		// 存储&a1, 相对%rsp偏移16
	movl	$23, 12(%rsp)		// 存储&a2, 相对%rsp偏移12
	movw	$34, 10(%rsp)		// 存储&a3, 相对%rsp偏移10
	movb	$45, 9(%rsp)		// 存储&a4, 相对%rsp偏移9

	leaq	12(%rsp), %rcx		// 寄存器存储变量4，即将&a2复制给%rcx
	leaq	16(%rsp), %rsi		// 寄存器存储变量2，即将&a1复制给%rsi

	leaq	9(%rsp), %rax
	pushq	%rax				// 参数8压入栈中，即: %rsp -= 8; (%rsp) = &a4
	pushq	$45					// 参数7压入占中，即: %rsp -= 8; (%rsp) = 45

	// 两次push之后，&a1相对%rsp偏移32；&a2相对%rsp偏移28；&a3相对%rsp偏移26；&a4相对%rsp偏移25

	leaq	26(%rsp), %r9		// 寄存器存储变量6，即将&a3复制给%r9
	movl	$34, %r8d			// 寄存器存储变量5，即将32复制给%r8d
	movl	$23, %edx			// 寄存器存储变量3，即将23复制给%edx
	movl	$12, %edi			// 寄存器存储变量1，即将12复制给%edi

	call	proc				// 将返回地址压入栈中，proc起始地址赋给PC, %rsp -= 8; (%rsp) = 返回地址; PC=起始地址

	movslq	28(%rsp), %rax		
	addq	32(%rsp), %rax		// a1 + a2
	movswl	26(%rsp), %edx
	movsbl	25(%rsp), %ecx		
	subl	%ecx, %edx			// a3 - a4
	movslq	%edx, %rdx

	imulq	%rdx, %rax			// (a1 + a2)*(a3 - a4)

	addq	$16, %rsp 			// 释放空间，对应两次push指令

	movq	24(%rsp), %rdi

	addq	$40, %rsp 			// 释放整个过程空间，对应第一条指令
	ret
------------------------------------------------------------------


2、包含被保存寄存器的运行时栈结构：
-----------------------------------------------------------------
long P(long x, long y)
{
	long u = Q(y);
	long v = Q(x);
	return u + v;
}
-----------------------------------------------------------------
P:
	pushq	%rbp 				// 被调用者保存寄存器
	pushq	%rbx 				// 被调用者保存寄存器
	subq	$8, %rsp 			// 分配栈空间，这里分配的空间作何使用，需要继续考究
	movq	%rdi, %rbp 			// 保存x到%rbp，否在第一次调用Q(y)时会将x值清除
	movq	%rsi, %rdi			// 保存y到%rsi，作为调用Q时的参数
	movl	$0, %eax			// 清空返回值寄存器
	call	Q@PLT				// 调用Q(y)
	movslq	%eax, %rbx 			// 将Q(y)返回值保存在%rbx中，否在在第二次调用Q(x)时，会将Q(y)的返回值清除
	movq	%rbp, %rdi			// 将x设置到参数寄存器%rsi中，此时%rbp的值为x
	movl	$0, %eax			// 清空返回值寄存器
	call	Q@PLT				// 调用Q(x)
	cltq
	addq	%rbx, %rax			// 求Q(x)+Q(y)，%rax保存着Q(x)的结果，%rbx保存着Q(y)的结果
	addq	$8, %rsp			// 释放栈空间
	popq	%rbx 				// 恢复被调用者保存寄存器的值，先进后出
	popq	%rbp
	ret 						// 将函数返回地址弹出，并设置到PC中，继续执行下一条指令
-----------------------------------------------------------------




