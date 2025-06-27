/**
 * @file lc039.cpp
 * @brief LeetCode 039 Combination Sum
 * @author sailing-innocent
 * @date 2025-02-18
 */


#include "test_util.h"
#include <vector>
#include <algorithm>
#include <functional>
#include <map>

namespace sail::test {

using std::vector;

struct CombineSumInput {
    vector<int> candidates;
    int target;
    vector<vector<int>> expected;
};

std::vector<CombineSumInput> combine_sum_inputs = {
    {{2, 3, 6, 7}, 7, {{2, 2, 3}, {7}}},
    {{2, 3, 5}, 8, {{2, 2, 2, 2}, {2, 3, 3}, {3, 5}}},
    {{2}, 1, {}},
    {{1}, 1, {{1}}},
    {{1}, 2, {{1, 1}}},
};

vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<vector<int>> res;
    // Sort 
    std::sort(candidates.begin(), candidates.end());
    vector<int> combination;

    // Recursive
    std::function<void(int, int)> backtrack = [&](int start, int remain) {
        if(remain == 0) {
            // fill, found one combination
            res.push_back(combination);
            return;
        }
        for(int i = start; i < candidates.size(); ++i) {
            if(candidates[i] > remain) {break;}
            combination.push_back(candidates[i]);
            backtrack(i, remain - candidates[i]);  // Not i+1 because we can reuse same element
            combination.pop_back();
        }
    };
    backtrack(0, target);
    return res;
}

}

TEST_CASE("lc_039") {
    using namespace sail::test;
    for(auto& input : combine_sum_inputs) {
        auto res = combinationSum(input.candidates, input.target);
        CHECK(res == input.expected);
    }
}