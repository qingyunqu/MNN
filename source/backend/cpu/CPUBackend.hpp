//
//  CPUBackend.hpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBackend_hpp
#define CPUBackend_hpp

#include <stdio.h>
#include <map>
#include <memory>
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class BufferAllocator;
class CPURuntime : public Runtime {
public:
    friend class CPUBackend;
    CPURuntime(const Backend::Info& info);
    virtual ~ CPURuntime();
    bool supportDot() const {
        return mIsSupportDot;
    }
    bool supportFp16() const {
        return mIsSupportFp16arith;
    }
    virtual Backend* onCreate() const override;
    virtual void onGabageCollect(int level) override;
    virtual float onGetMemoryInMB() override;
//private:
    std::shared_ptr<BufferAllocator> mStaticAllocator;
    int mThreadNumber;
    int mTaskIndex;
    size_t mFlags;
    BackendConfig::MemoryMode mMemory;
    BackendConfig::PowerMode mPower;
    BackendConfig::PrecisionMode mPrecision;

    // Backend features
    // CPU features
    bool mIsSupportDot = false;
    bool mIsSupportFp16arith = false;
    float mFlops = 0.0f;
    static Backend*(*gExtraCreate)(const Runtime* runtime);
};

class CPUBackend : public Backend {
public:
    CPUBackend(const CPURuntime* runtime, MNNForwardType type = MNN_FORWARD_CPU);
    virtual ~CPUBackend();

    // Return sizeDivide, scheduleNumber aligned memory
    std::pair<int, int> multiThreadDivide(int size) const;
public:
    virtual bool onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) override;
    virtual bool onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual bool onRequireBufferFromOS(const Tensor* nativeTensor) override;
    virtual bool onFreeBufferToOS(const Tensor* nativeTensor) override;
    virtual bool onRequireBufferHybrid(const Tensor* nativeTensor, int hybrid_thres=MNN_HYBRID_DYNAMIC_THRESHOLD) override;
    virtual bool onFreeBufferHybrid(const Tensor* nativeTensor, int hybrid_thres=MNN_HYBRID_DYNAMIC_THRESHOLD) override;
    virtual void changeBufferType(BufferType bufferType) override;
    virtual void setHeuristicStrategy(bool flag) override;
    virtual void configHeuristicStrategy(std::string modelName, int batchsize) override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;
    virtual std::pair<float, bool> onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                            const MNN::Op* op) override;

    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;
    
public:
    class Creator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };

    static bool addCreator(OpType t, Creator* c);

    int threadNumber() const {
        return mRuntime->mThreadNumber;
    }

    BufferAllocator* getBufferAllocator() const {
        return mDynamicAllocator.get();
    }

    BackendConfig::MemoryMode memoryMode() const {
        return mRuntime->mMemory;
    }
#ifdef MNN_USE_THREAD_POOL
    inline int taskIndex() const {return mRuntime->mTaskIndex;}
#endif
    bool supportDot() const;
    static void initCreatorMap();

protected:
    bool allocBuffer(int size, Tensor* dest,  StorageType storageType);
private:
    std::shared_ptr<BufferAllocator> mStaticAllocator;
    std::shared_ptr<BufferAllocator> mDynamicAllocator;
    bool mCheckNAN = false;
    const CPURuntime* mRuntime;
    static std::map<OpType, CPUBackend::Creator*>* getCreatorMap();
    static std::map<OpType, CPUBackend::Creator*>* gCreator;
    int mDynamicOutputCacheID = -1;
    int mDynamicResizeID = -1;
    bool mHeuristic = false;
};

#define REGISTER_CPU_OP_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {            \
        static name _temp;\
        CPUBackend::addCreator(opType, &_temp); \
    }

} // namespace MNN

#endif /* CPUBackend_hpp */
