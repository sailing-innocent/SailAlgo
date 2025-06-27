/**
 * @file date.cpp
 * @brief The date related utilities.
 * @author sailing-innocent
 * @date 2025-02-18
 */

#include "util/date.h"
#include "test_util.h"

namespace sail::util {

int dayofweek(int y, int m, int d) {
    y -= m < 3;
    // static int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    // return (y + y / 4 - y / 100 + y / 400 + t[m - 1] + d) % 7;
    return (y + y/4 - y/100 + y/400 + "-bed=pen+mad."[m] + d) % 7;
}

}// namespace sail::util


TEST_SUITE("date") {
    using namespace sail::util;
    TEST_CASE("dayofweek") {
        CHECK(dayofweek(2025, 2, 18) == 2);
    }
}