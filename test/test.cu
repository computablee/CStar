#include <cstar.cu>
#include <iostream>

using shape = Shape<1024>;

int main()
{
    shape::shape<int> myshape1, myshape2;
    int sum = 0;

    myshape1 = 3;
    myshape2 = 4;

    myshape1 += myshape2;

    sum += myshape1;

    std::cout << sum << std::endl;
    std::cout << 1024 * 7 << std::endl;
}