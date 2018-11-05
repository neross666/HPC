

// long last(long u, long v)
// {
// 	return u*v;
// }

// long first(long x)
// {
// 	return last(x-1, x+1);
// }

// int main()
// {
// 	long z = first(10);
// 	return z;
// }


// int proc(int a, short b, long* u, char* v)
// {
// 	*u += a;
// 	*v += b;
// 	return sizeof(a)+sizeof(b);
// }


// long swap_add(long *xp, long *yp)
// {
// 	long x = *xp;
// 	long y = *yp;
// 	*xp = y;
// 	*yp = x;
// 	return x + y;
// }

// long caller()
// {
// 	long arg1 = 534;
// 	long arg2 = 1057;
// 	long sum = swap_add(&arg1, &arg2);
// 	long diff = arg1 - arg2;
// 	return sum*diff;
// }



// void proc(long a1, long *a1p, 
// 	int a2, int* a2p,
// 	short a3, short* a3p,
// 	char a4, char* a4p)
// {
// 	*a1p += a1;
// 	*a2p += a2;
// 	*a3p += a3;
// 	*a4p += a4;
// }
// long multi()
// {
// 	long a1=12;
// 	int a2=23;
// 	short a3=34;
// 	char a4=45;
// 	proc(a1, &a1, a2, &a2, a3, &a3, a4, &a4);
// 	return (a1+a2)*(a3-a4);
// }



// long call_proc()
// {
// 	long x1=1; int x2=2;
// 	short x3=3; char x4=4;
// 	proc(x1, &x1, x2, &x2, x3, &x3, x4, &x4);
// 	return (x1+x2)*(x3-x4);
// }

// long P(long x)
// {
// 	long a0 = x;
// 	long a1 = x+1;
// 	long a2 = x+2;
// 	long a3 = x+3;
// 	long a4 = x+4;
// 	long a5 = x+5;
// 	long a6 = x+6;
// 	long a7 = x+7;
// 	Q();

// 	return a0+a1+a2+a3+a4+a5+a6+a7;
// }


// long rfact(long n)
// {
// 	long result;
// 	if (n <= 1)
// 	{
// 		result = 1;
// 	}else
// 	{
// 		result = n*rfact(n-1);
// 	}
// 	return result;
// }


// long rfun(unsigned long x)
// {
// 	if (x == 0)
// 	{
// 		return 0;
// 	}
// 	unsigned long nx = x>>2;
// 	long rv = rfun(nx);
// 	return x+rv;
// }


// #include <stdio.h>
// int main()
// {
// 	short** U[6];
// 	printf("%ld\n", sizeof(U));

// 	return 0;
// }

// #define M 5
// #define N 7

// long P[M][N];
// long Q[N][M];

// long sum_elem(long i, long j)
// {
// 	return P[i][j] + Q[j][i];
// }


// #define N 16
// typedef int fix_matrix[N][N];

// int fix_prod_ele_opt(fix_matrix A, int val)
// {
// 	long i;

// 	for (int i = 0; i < N; ++i)
// 	{
// 		A[i][i] = val;
// 	}

// 	return result;
// }


// int var_ele(long n, int A[n][n], long i, long j)
// {
// 	return A[i][j];
// }

// int main(int argc, char const *argv[])
// {
// 	int n = 12;
// 	int A[n][n];
// 	return 0;
// }


// struct prob
// {
// 	int* p;
// 	struct
// 	{
// 		int x;
// 		int y;
// 	} s;
// 	struct prob* next;
// };

// void sp_init(struct prob *sp)
// {
// 	sp->s.x = sp->s.y;
// 	sp->p = &sp->s.x;
// 	sp->next = sp;
// }


// struct ELE
// {
// 	long v;
// 	struct ELE* p;
// };

// long fun(struct ELE* ptr)
// {
// 	long ret = 0;
// 	while(ptr != 0)
// 	{
// 		ret += ptr->v;
// 		ptr = ptr->p;
// 	}
// 	return ret;
// }

// typedef union {
// 	struct{
// 		long u; 
// 		short v;
// 		char w;
// 	}t1;
// 	struct 
// 	{
// 		int a[2];
// 		char* p;
// 	}t2;
// }u_type;

// void get(u_type* up, char* dest)
// {
// 	*dest = *up->t2.p;
// }


// #include <stdio.h>
// int main(int argc, char const *argv[])
// {
// 	typedef struct P1
// 	{
// 		char* a;
// 		short b;
// 		double c;
// 		char d;
// 		float e;
// 		char f;
// 		long g;
// 		int h;
// 	}P1;

// 	int ss = sizeof(P1);
// 	printf("%d\n", ss);
// 	return 0;
// }


// char echo()
// {
// 	char buf[8];
// 	return add(buf[0], buf[1]) + sub(buf[0], buf[1]);
// }


// #include <stdio.h>

// char* gets(char* s)
// {
// 	int c;
// 	char *dest = s;
// 	while ((c == getchar()) != '\n' && c != EOF)
// 		*dest++ = c;

// 	if (c == EOF && dest == s)
// 	{
// 		return 0;
// 	}
// 	*dest++ = '\0';

// 	return s;
// }

// void echo()
// {
// 	char buf[8];
// 	gets(buf);
// 	puts(buf);
// }

// char* get_line()
// {
// 	char buf[4];
// 	char* result;
// 	gets(buf);
// 	result = malloc(strlen(buf));
// 	strcpy(result, buf);
// 	return result;
// }


// float func(float v1, float* src, float* dst)
// {
// 	float v2 = *src;
// 	*dst = v1;
// 	int si = (int)v2;
// 	return (float)si;
// }


// double funct(double a, float x, double b, int i)
// {
// 	return a*x - b/i;
// }

#define PI 3.1415926
#define N 3


typedef struct Cicle
{
	double radius;
	long pt_x;
	long pt_y;
};

double CalculateArea(Cicle* cicle)
{
	return 2*PI*(cicle->radius)*(cicle->radius);
}

void Total()
{
	Cicle cicles[N][N];
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			cicle.radius = rand();
			cicle.pt_x = ;
			cicle.pt_y = ;
		}
	}
}