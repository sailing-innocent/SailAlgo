/**
  * @file quick_sort.cpp
  * @brief the quick sort algorithm
  * @author sailing-innocent
  * @date 2024-07-29
  */

#include "test_util.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

namespace sail::test {

void quick_sort_impl(std::span<int> unsorted, int left, int right) {
	int curr = left;
	for (auto i = left; i <= right; i++) {
		if (unsorted[i] < unsorted[right]) {
			std::swap(unsorted[i], unsorted[curr]);
			curr++;
		}
	}
	std::swap(unsorted[curr], unsorted[right]);
	if (curr - left > 1) {
		quick_sort_impl(unsorted, left, curr - 1);
	}
	if (right - curr > 1) {
		quick_sort_impl(unsorted, curr + 1, right);
	}
}

int quick_sort(std::span<int> unsorted, int N) {
	// std::sort(unsorted.begin(), unsorted.end());
	quick_sort_impl(unsorted, 0, N - 1);
	return 0;
}

}// namespace sail::test

TEST_CASE("quick_sort") {
	constexpr int N = 20;
	std::vector<int> unsorted(N);
	std::iota(unsorted.begin(), unsorted.end(), 0);
	// shuffle
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(unsorted.begin(), unsorted.end(), g);

	std::vector<int> sorted(N);
	std::iota(sorted.begin(), sorted.end(), 0);
	sail::test::quick_sort(unsorted, N);

	CHECK(sail::test::int_span_equal<int>(unsorted, sorted));

	CHECK(0 == 0);
}