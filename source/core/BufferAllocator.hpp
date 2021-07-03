//
//  BufferAllocator.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BufferAllocator_hpp
#define BufferAllocator_hpp

#include <map>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include "MNNMemoryUtils.h"
#include "NonCopyable.hpp"
#define TEST() printf("test\n");

namespace MNN {

/** memory utils wrapper. provides memory reusing with alignment ability. */
class MNN_PUBLIC BufferAllocator : public NonCopyable {
public:
    class Allocator {
    public:
        Allocator() = default;
        virtual ~ Allocator() = default;
        virtual std::pair<void*, int> onAlloc(int size) = 0;
        virtual void onRelease(std::pair<void*, int> ptr) = 0;
        static std::shared_ptr<Allocator> createDefault();
        static std::shared_ptr<Allocator> createRecurse(BufferAllocator* parent);
    };
    /**
     * @brief init buffer allocator with pointer alignment.
     * @param align given pointer alignment.
     */
    BufferAllocator(std::shared_ptr<Allocator> parent, int align = MNN_MEMORY_ALIGN_DEFAULT) : mAllocator(parent), mAlign(align) {
        // nothing to do
    }
    /**
     * @brief deinit buffer allocator. frees all allocated memories.
     */
    ~BufferAllocator() {
//        MNN_PRINT("%s: call %s\n",mName.c_str(), __FUNCTION__ )
        release();
    }

public:
    /**
     * @brief alloc CHUNK pointer with given size. if any reusable pointer matches size, reuse it.
     * @param size  given size.
     * @param seperate if true, the memory can't be alloc from free pool
     * @return allocated or used CHUNK pointer.
     * @sa free
     * @sa release
     */
    std::pair<void*, int> alloc(int size, bool seperate = false);

    std::pair<void*, int> allocHeuristically(std::string id, int size);
    bool freeHeuristically(std::string id, std::pair<void*, int> pointer);
    static std::pair<void*, int> allocaFromOS(int size);
    static bool freeToOS(std::pair<void*, int>);

    /**
     * @brief mark CHUNK pointer as reusable.
     * @param pointer   given CHUNK pointer.
     * @return true if pointer is a CHUNK pointer, false otherwise.
     * @sa release
     */
    bool free(std::pair<void*, int> pointer);

    /**
     * @brief free all allocated memories.
     * @sa allocSeparate
     * @sa alloc
     * if allRelease, clear all memory , otherwise delete freelist
     */
    void release(bool allRelease = true);

    /**
     * @brief query total size allocated indeed.
     * @return total size allocated indeed.
     */
    size_t totalSize() const {
        return mTotalSize;
    }

    void debugUsage();

    /*
     For multi thread case,
     we must assume that the memory use by different thread don't conflict
     begin barrier / end barrier means enter the alloc for multi-thread
     begin group / end group means the memory allocated belong to one thread
     different group must use different memory,
     but the origin freelist can be used by every group
     */
    void barrierBegin();
    void barrierEnd();
    void beginGroup();
    void endGroup();
    void setName(std::string name) {
        mName = std::move(name);
    }
    void setHeuristicStrategy(std::string model, int batch);

private:
    // MNN通过node组织成了一个树形结构，然后只保留叶子结点的索引，叶子结点通过parent指针向上找到根结点
    // 修改后通过双向链表的形式组织，增加了合并可用内存的可能性
    class Node {
    public:
        ~Node();
        std::pair<void*, int> pointer;
        std::shared_ptr<Node> parent = nullptr;
        std::shared_ptr<Node> left = nullptr, right = nullptr;
        int32_t size;
        int16_t useCount = 0;
        // 每个node至多被拆分成2个node，一个used一个freed（尽管freed可能被继续拆分）
        // usecount表示当前node是否被拆分出来一个used + freed是否被拆分成了used
        // 等价于左边used是否被继续使用了 + 右边freed是否被继续使用了
        // 等价于左右【直接指向自己】的child节点中有多少被使用了
        Allocator* outside = nullptr;
    };

    typedef std::multimap<size_t, std::shared_ptr<Node>> FREELIST;

    void returnMemory(FREELIST* list, std::shared_ptr<Node> node, bool permitMerge = true);
    std::pair<void*, int> getFromFreeList(FREELIST* list, int size, bool permiteSplit = true);

    std::map<std::pair<void*, int>, std::shared_ptr<Node>> mUsedList;
    FREELIST mFreeList;
    size_t mTotalSize   = 0;

    FREELIST* mCurrentFreeList = nullptr;
    std::vector<std::shared_ptr<FREELIST>> mGroups;
    std::shared_ptr<Allocator> mAllocator;
    int mAlign;
    std::string mName = "static";
    std::map<std::string, size_t> mHeuristicStrategy;
    void* mHeuristicPtr;
    size_t mHeuristicSize;
};
} // namespace MNN
#endif
