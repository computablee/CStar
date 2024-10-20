#include <cstar.cu>
#include <iostream>

using shape = Shape<17, 413>;

int main()
{
    shape::shape<double> myshape1, myshape2;
    double sum = 0;

    myshape1 = 3.14;
    myshape2 = 4.0;

    myshape1 += myshape2;

    sum += myshape1;

    std::cout << sum << std::endl;
    std::cout << 17 * 413 * 7.14 << std::endl;
}