#include <iostream>
#include <cstdio>
#include <vector>
using namespace std;

int main()
{
  std::cout << "Hello Git" << std::endl;
  int iarr[]{0,1,3,5};
  vector<int> vec(iarr,iarr+sizeof(iarr)/sizeof(int));
  for(vector<int>::iterator it = begin(vec);vec!=end(vec);i++)
  {
	  printf("Number %d is %d\n",distance(begin(vec),it),*it);
  }
  return 0;
}