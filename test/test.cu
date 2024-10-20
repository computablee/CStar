#include <cstar.cu>
#include <iostream>

using namespace CStar;

using shape = Shape<17, 413>;

int main()
{
    shape::shape<float> myshape1, myshape2;
    float sum = 0;

    myshape1 = 3.14;
    myshape2 = 4.0;

    myshape1 += myshape2;

    sum += myshape1;

    std::cout << sum << std::endl;
    std::cout << 17 * 413 * 7.14 << std::endl;

    std::cout << myshape1(16, 412) << std::endl;

    myshape2(4, 6) = 16.8f;

    std::cout << myshape2(4, 6) << std::endl;
    std::cout << myshape2(4, 7) << std::endl;

    std::cout << "Rank of shape is " << rankof_t<shape> << std::endl;
    std::cout << "Positions of shape is " << positionsof_t<shape> << std::endl;
}