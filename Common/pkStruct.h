#ifndef PK_STRUCT_H
#define PK_STRUCT_H
#include<vector>
namespace pk
{
	template<typename _Tp> struct _tDLparam
	{
		_tDLparam():activeType(0),pred(false),predData(NULL){};
		_tDLparam(int activeType):activeType(activeType),predData(NULL){};
		_tDLparam(std::vector<_Tp> labels):activeType(activeType),predData(NULL)
		{
			this->labels=labels;
		};

		~_tDLparam()
		{
			if(predData!=NULL)
				delete [] predData;
		}
		int activeType;
		std::vector<_Tp> labels;
		bool pred;
		double* predData;
	};
}

typedef pk::_tDLparam<int> DLparam;

#endif