#include "test_util.h"
/**
  * @file lc048.cpp
  * @brief the leetcode 048 rotate image
  * @author sailing-innocent
  * @date 2024-07-29
  */

#include <vector>
#include <span>

namespace sail::test {

bool matrix_equal(std::vector<std::vector<int>>& mat1, std::vector<std::vector<int>>& mat2, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			CHECK(mat1[i][j] == mat2[i][j]);
			if (mat1[i][j] != mat2[i][j]) {
				return false;
			}
		}
	}
	return true;
}

void rotate(std::vector<std::vector<int>>& mat) {
	int M = mat.size();
	int N = mat[0].size();
	// flip the matrix according to the diagonal
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			if (i < j) {
				std::swap(mat[i][j], mat[j][i]);
			}
		}
	}
	// flip the matrix according to y mid
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N / 2; j++) {
			std::swap(mat[i][j], mat[i][N - j - 1]);
		}
	}
}

int test_rotate_image() {
	std::vector<std::vector<int>> matrix = {
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9}};
	std::vector<std::vector<int>> expected = {
		{7, 4, 1},
		{8, 5, 2},
		{9, 6, 3}};

	rotate(matrix);

	CHECK(matrix_equal(matrix, expected, 3, 3));

	return 0;
}

}// namespace sail::test

TEST_CASE("lc::048") {
	using namespace sail::test;
	CHECK(test_rotate_image() == 0);
}