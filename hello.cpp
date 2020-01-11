#include <iostream>
#include <algorithm>
#include <cstdio>
#include <vector>
using namespace std;

int main()
{
  std::cout << "Hello Git" << std::endl;
  int iarr[]{0,1,6,5};
  vector<int> vec(iarr,iarr+sizeof(iarr)/sizeof(int));
  for(vector<int>::iterator it = begin(vec);vec!=end(vec);it++)
  {
	  printf("Number %d is %d\n",distance(begin(vec),it),*it);
  }
  sort(vec.begin(),vec.end(),greater<int>());
  it = find_if(vec.begin(),vec.end(),[](int x){return x%2!=0;});
  if(it!=vec.end())
	  printf("Index : %d , Data : %d\n",distance(vec.begin(),it),*it);
  return 0;
}