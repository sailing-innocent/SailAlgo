#pragma once
/**
 * @file test_util.h
 * @brief The Test Framework
 * @author sailing-innocent
 * @date 2024-06-17
 */

#include <doctest.h>
#include <span>
#include <concepts>

namespace sail::test {

[[nodiscard]] int argc() noexcept;
[[nodiscard]] const char* const* argv() noexcept;
[[nodiscard]] bool float_span_equal(std::span<float> a, std::span<float> b);
template<std::integral T>
[[nodiscard]] bool int_span_equal(std::span<T> a, std::span<T> b) {
	CHECK(a.size() == b.size());
	if (a.size() != b.size()) {
		return false;
	}
	for (size_t i = 0; i < a.size(); i++) {
		CHECK(a[i] == b[i]);
		if (a[i] != b[i]) {
			return false;
		}
	}
	return true;
}

}// namespace sail::test