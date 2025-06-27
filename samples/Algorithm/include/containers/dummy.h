#pragma once 
/**
 * @file dummy.h
 * @brief The Dummy Containers
 * @author sailing-innocent
 * @date 2025-02-18
 */

namespace sail::dummy {

template<typename T>
struct ListNode {
    T val;
    ListNode* next;
    ListNode()
        : val(0)
        , next(nullptr)
    {
    }

    ListNode(T x)
        : val(x)
        , next(nullptr)
    {
    }

    ListNode(T x, ListNode* next)
        : val(x)
        , next(next)
    {
    }
};


}