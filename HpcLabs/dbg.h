#pragma once

#define _MEMCHKE

#ifdef _DEBUG

#ifdef _MEMCHKE
#define new new(_CLIENT_BLOCK, __FILE__, __LINE__)
#endif

#endif // _DEBUG