#include <cstar.cu>
#include <iostream>
#include <utility>
#include <cassert>
#include <cmath>

using namespace CStar;

using shape = Shape<17, 413>;
using shape2 = Shape<20, 20>;

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

    auto myshape4 = myshape2 * 2;

    assert(myshape4(4, 6) == 16.8f * 2);
    assert(myshape4(5, 8) == 8.0f);

    auto myshape5 = std::move(myshape4);

    shape2::shape<float> mult_reduce = 1.01f;
    float mult_reduce_prod = 2;

    mult_reduce_prod *= mult_reduce;
    auto expected_mult_reduce = pow(1.01f, 400) * 2;

    assert(!(mult_reduce_prod < expected_mult_reduce - 0.1f || mult_reduce_prod > expected_mult_reduce + 0.1f));

    myshape1(4, 5) = 15;
    myshape1(0, 0) = myshape1(4, 5);
    assert(myshape1(0, 0) == 15);

    std::cout << "Tests passed." << std::endl;
}