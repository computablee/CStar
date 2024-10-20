#include <cstar.cu>
#include <iostream>

using shape = Shape<1024>;

int main()
{
    shape::shape<int> myshape1, myshape2;

    myshape1 = 3;
    myshape2 = 4;

    myshape1 += myshape2;

    std::cout << myshape1[513] << std::endl;
}