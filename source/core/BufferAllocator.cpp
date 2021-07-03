//
//  BufferAllocator.cpp
//  MNN
//
//  Created by MNN on 2018/12/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/BufferAllocator.hpp"
#include "core/Macro.h"

//#define DEBUG_EXECUTION_DETAIL

#ifdef DEBUG_EXECUTION_DETAIL
#define MNN_DEBUG_PRINT(format, ...) MNN_PRINT(format, ##__VA_ARGS__)
#else
#define MNN_DEBUG_PRINT(format, ...)
#endif
//#define DUMP_USAGE
//#define MNN_DEBUG_MEMORY
namespace MNN {
class DefaultAllocator : public BufferAllocator::Allocator {
public:
    DefaultAllocator() {
        // Do nothing
    }
    virtual ~ DefaultAllocator() {
        // Do nothing
    }
    virtual std::pair<void*, int> onAlloc(int size) {
        MNN_DEBUG_PRINT("\tfail to reuse memory, require from OS with %d byte\n", size)
        return std::make_pair(MNNMemoryAllocAlign(size, MNN_MEMORY_ALIGN_DEFAULT), 0);
    }
    virtual void onRelease(std::pair<void*, int> ptr) {
        MNN_ASSERT(ptr.second == 0);
        MNNMemoryFreeAlign(ptr.first);
    }
};
class RecurseAllocator : public BufferAllocator::Allocator {
public:
    RecurseAllocator(BufferAllocator* parent) {
        mParent = parent;
    }
    virtual ~ RecurseAllocator() {
        // Do nothing
    }
    virtual std::pair<void*, int> onAlloc(int size) override {
        return mParent->alloc(size);
    }
    virtual void onRelease(std::pair<void*, int> ptr) override {
        mParent->free(ptr);
    }
private:
    BufferAllocator* mParent;
};

std::shared_ptr<BufferAllocator::Allocator> BufferAllocator::Allocator::createDefault() {
    std::shared_ptr<BufferAllocator::Allocator> _res;
    _res.reset(new DefaultAllocator);
    return _res;
}
std::shared_ptr<BufferAllocator::Allocator> BufferAllocator::Allocator::createRecurse(BufferAllocator* parent) {
    std::shared_ptr<BufferAllocator::Allocator> _res;
    _res.reset(new RecurseAllocator(parent));
    return _res;
}

BufferAllocator::Node::~Node() {
    MNN_DEBUG_PRINT("call %s\n", __FUNCTION__ )
    if (nullptr == parent) {
        outside->onRelease(pointer);
    }
}
std::pair<void*, int> BufferAllocator::alloc(int size, bool seperate) {
    MNN_DEBUG_PRINT("\t%s: call %s\n", mName.c_str(), __FUNCTION__ )
#ifdef DUMP_USAGE
    auto memoryUsed = size / 1024.0f / 1024.0f;
    MNN_PRINT("Alloc: %f\n", memoryUsed);
#endif
    std::pair<void*, int> pointer;
    // reuse if possible
    if (!seperate) {
        if (nullptr != mCurrentFreeList) {
            pointer = getFromFreeList(mCurrentFreeList, size, false);
        }
        if (nullptr != pointer.first) {
            return pointer;
        }
        if (mName == "dynamic" && mCurrentFreeList == nullptr) {
            MNN_DEBUG_PRINT("try get %d bytes dynamic memory \n", UP_DIV(size, mAlign) * mAlign)
        }
        pointer = getFromFreeList(&mFreeList, size);
        if (nullptr != pointer.first) {
            return pointer;
        }
    }
    MNN_DEBUG_PRINT("\t%s: fail to get from free list, allocate otherwise\n", mName.c_str());

    // alloc otherwise
    pointer = mAllocator->onAlloc(size);
    if (nullptr == pointer.first) {
        return pointer;
    }
    mTotalSize += size;

    // save node
    std::shared_ptr<Node> node(new Node);
    node->size         = size;
    node->pointer      = pointer;
    mUsedList[pointer] = node;
    node->outside      = mAllocator.get();

#ifdef DUMP_USAGE
    MNN_PRINT("mTotalSize: %f\n", mTotalSize / 1024.0f / 1024.0f);
#endif
    return pointer;
}

void BufferAllocator::setHeuristicStrategy(std::string model, int batch) {
    release();
    char filename[50];
    sprintf(filename, "%s.%d.txt", model.c_str(), batch);
    MNN_DEBUG_PRINT("%s: %s: filename = %s\n", mName.c_str(), __FUNCTION__ , filename)
    std::ifstream ifs(filename, std::ios::in);
    std::string s;
    size_t a;
    while (ifs >> s >> a) {
        if (s == "maxsize") {
            mHeuristicSize = a;
        } else {
            mHeuristicStrategy[s] = a;
        }
    }
    ifs.close();
    MNN_DEBUG_PRINT("%s: %s: maxsize = %d\n", mName.c_str(), __FUNCTION__ , mHeuristicSize)
    auto heuristicPool = alloc(mHeuristicSize);
    mHeuristicPtr = heuristicPool.first;
}

std::pair<void*, int> BufferAllocator::allocHeuristically(std::string id, int size) {
    MNN_DEBUG_PRINT("\t%s: call %s\n",mName.c_str(), __FUNCTION__ )
    debugUsage();
    if (mHeuristicStrategy.empty()) {
        return alloc(size, false);
    }
    if (mHeuristicStrategy.find(id) == mHeuristicStrategy.end()) {
        return alloc(size, false);
    }

    return std::make_pair(mHeuristicPtr, mHeuristicStrategy[id]);
}

bool BufferAllocator::freeHeuristically(std::string id, std::pair<void *, int> pointer) {
    MNN_DEBUG_PRINT("\t call %s\n", __FUNCTION__ )
    if (mHeuristicStrategy.empty()) {
        return free(pointer);
    }
    if (mHeuristicStrategy.find(id) == mHeuristicStrategy.end()) {
        return free(pointer);
    }
    return true;
}

std::pair<void*, int> BufferAllocator::allocaFromOS(int size) {
    return std::make_pair(MNNMemoryAllocAlign(size, MNN_MEMORY_ALIGN_DEFAULT), 0);
}

bool BufferAllocator::freeToOS(std::pair<void *, int> ptr) {
    MNN_ASSERT(ptr.second == 0);
    MNNMemoryFreeAlign(ptr.first);
    return true;
}

void BufferAllocator::returnMemory(FREELIST* listP, std::shared_ptr<Node> node, bool permitMerge) {
//    MNN_PRINT("\tcall %s: %s\n", mName.c_str(), __FUNCTION__ )
    if (*listP == mFreeList && mName == "dynamic") {
        MNN_DEBUG_PRINT("try return %d bytes to freelist\n", node->size)
    }

    auto& list = *listP;
    list.insert(std::make_pair(node->size, node));
    // update parent use count
    if (nullptr != node->parent && permitMerge) {
        auto parent = node->parent;
        parent->useCount -= 1;

        // merge if all subnodes were freed
        auto needMerge = parent->useCount == 0;
        while (needMerge) {
            // collect all subnodes
            for (auto iter = list.begin(); iter != list.end();) {
                if (iter->second->parent == parent) {
                    iter = list.erase(iter);
                    continue;
                }
                iter++;
            }

            // do merge downside up
            list.insert(std::make_pair(parent->size, parent));
            needMerge = false;
            if (parent->parent != nullptr) {
                parent = parent->parent;
                parent->useCount -= 1;
                needMerge = parent->useCount == 0;
            }
        }
    }
}

bool BufferAllocator::free(std::pair<void*, int> pointer) {
    // get node
    MNN_DEBUG_PRINT("\t%s: call %s\n", mName.c_str(), __FUNCTION__ )
    auto x = mUsedList.find(pointer);
    if (x == mUsedList.end()) {
        MNN_ASSERT(false);
        return false;
    }
    // mark as reusable
    auto node = x->second;
    mUsedList.erase(x);
    if (nullptr != mCurrentFreeList) {
        returnMemory(mCurrentFreeList, node, false);
    } else {
        returnMemory(&mFreeList, node);
    }
    debugUsage();
#ifdef DUMP_USAGE
    auto memoryUsed = x->second->size / 1024.0f / 1024.0f;
    MNN_PRINT("Free: %f\n", memoryUsed);
#endif
    return true;
}

void BufferAllocator::debugUsage() {
    size_t used_size = 0, free_size = 0;
    for(auto iter: mUsedList) {
        used_size += iter.second->size;
    }
    for (auto iter: mFreeList) {
        free_size += iter.first;
    }
    MNN_DEBUG_PRINT("%s: %s: mUsedList.size = %d with memsize = %d, mFreeList.size = %d with memsize = %d\n",
                    mName.c_str(), __FUNCTION__ , mUsedList.size(), used_size, mFreeList.size(), free_size)
}

void BufferAllocator::release(bool allRelease) {
    MNN_DEBUG_PRINT("%s: %s: %d\n", mName.c_str(), __FUNCTION__ , allRelease);
    MNN_ASSERT(mGroups.empty());
    debugUsage();
    if (allRelease) {
        mUsedList.clear();
        mFreeList.clear();
        mTotalSize = 0;
        return;
    }
    for (auto f : mFreeList) {
        if (f.second->parent == nullptr) {
            MNN_ASSERT(mTotalSize >= f.first);
            mTotalSize -= f.first;
        }
    }
    mFreeList.clear();
}

void BufferAllocator::barrierBegin() {
    MNN_ASSERT(mGroups.empty());
}

void BufferAllocator::barrierEnd() {
    for (auto& freeGroup : mGroups) {
        auto freeList = *freeGroup;
        for (auto& iter : freeList) {
            returnMemory(&mFreeList, iter.second);
        }
    }
    mGroups.clear();
}

void BufferAllocator::beginGroup() {
    std::shared_ptr<FREELIST> newFreeList(new FREELIST);
    mCurrentFreeList = newFreeList.get();
    mGroups.emplace_back(newFreeList);
}

void BufferAllocator::endGroup() {
    mCurrentFreeList = nullptr;
}

std::pair<void*, int> BufferAllocator::getFromFreeList(FREELIST* list, int size, bool permiteSplit) {
//    MNN_PRINT("\t%s: call %s\n", mName.c_str(), __FUNCTION__ )
#ifdef MNN_DEBUG_MEMORY
    return std::make_pair(nullptr, 0);
#endif

    // get node larger than size
    int tot_size = 0;
    for (auto iter: *list) {
        tot_size += iter.first;
    }
    MNN_DEBUG_PRINT("\t%s: free list.size = %d, tot_size = %d\n", mName.c_str(), list->size(), tot_size)
    auto x = list->lower_bound(size);
    if (x == list->end()) {
        if (list->size()) {
            auto max_size = std::max_element(
                        list->begin(), list->end(),
                        [](const std::pair<size_t, std::shared_ptr<Node>>& p1, const std::pair<size_t, std::shared_ptr<Node>>& p2){
                            return p1.first < p2.first;
                        }
                    )->first;
            MNN_DEBUG_PRINT("\t%s: free list is not empty with max_size = %d, but fail to match size = %d\n", mName.c_str(), int(max_size), size)
        }
        return std::make_pair(nullptr, 0);
    }
    MNN_DEBUG_PRINT("\t%s: match size %d with %d\n", mName.c_str(), size, int(x->first))

    // update parent use count
    auto pointer = x->second->pointer;
    if (permiteSplit && nullptr != x->second->parent) {
        x->second->parent->useCount += 1;
    }

    // uses up all aligned space
    auto sizeAlign = UP_DIV(size, mAlign) * mAlign;
    if (sizeAlign >= x->first || (!permiteSplit)) {
        mUsedList.insert(std::make_pair(pointer, x->second));
        list->erase(x);
        return pointer;
    }

    // split otherwise
    std::shared_ptr<Node> first(new Node);
    first->parent  = x->second;
    first->size    = sizeAlign;
    first->pointer = x->second->pointer;
    first->outside = mAllocator.get();
    mUsedList.insert(std::make_pair(pointer, first));
    x->second->useCount += 1;

    std::shared_ptr<Node> second(new Node);
    second->outside = mAllocator.get();
    second->parent  = x->second;
    second->size    = x->second->size - sizeAlign;
    second->pointer.first = x->second->pointer.first;
    second->pointer.second = x->second->pointer.second + sizeAlign;
    list->erase(x);
    list->insert(std::make_pair(second->size, second));
    return pointer;
}
} // namespace MNN
