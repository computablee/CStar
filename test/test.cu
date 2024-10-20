#include <cstar.cu>
#include <iostream>
#include <cassert>

using namespace CStar;

using shape = Shape<17, 413>;

int main()
{
    shape::shape<float> myshape1;
    float sum = 0;

    myshape1 = 3.14;
    shape::shape<float> myshape2 = 4.0f;

    auto myshape3 = myshape1 + myshape2;

    sum += myshape3;
    auto expected_sum = 17 * 413 * 7.14f;

    assert(!(sum < expected_sum - 0.1 || sum > expected_sum + 0.1));
    assert(myshape1(16, 412) == 3.14f);

    myshape2(4, 6) = 16.8f;
    assert(myshape2(4, 6) == 16.8f);
    assert(myshape2(4, 7) == 4.0f);

    assert(rankof_t<shape> == 2);
    assert(positionsof_t<shape> == 17 * 413);

    std::cout << "Tests passed." << std::endl;
}