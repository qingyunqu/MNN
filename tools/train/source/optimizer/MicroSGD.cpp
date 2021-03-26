//
//  MicroSGD.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MicroSGD.hpp"
#include "OpGrad.hpp"
using namespace MNN::Express;


namespace MNN {
namespace Train {
MicroSGD::MicroSGD(std::shared_ptr<Module> module, int batchSize, int microBatchSize)
        : SGD(module), mBatchSize(batchSize), mMicroBatchSize(microBatchSize){
    MNN_ASSERT(mBatchSize % mMicroBatchSize == 0)
    auto train = ParameterOptimizer::trainable();
    for (auto p : train) {
        mHistory[p] = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
    }
}

bool MicroSGD::step(Express::VARP loss) {
    mMicroStep++;
    printf("call step in MicroSGD, mMicroStep = %d, mStep = %d\n", mMicroStep, currentStep());
    if (mBatchSize == mMicroBatchSize) {
        printf("don't need to use the micro-batch tech, call father's step directly\n");
        return SGD::step(loss);
    }
    auto res = this->onGetNextParameter(loss);
    if (res.empty()) {
        return false;
    }
    if (mMicroStep % (mBatchSize / mMicroBatchSize) == 1) {
        //start a new batch, reset gradBuffer
        printf("start a new batch, reset gradBuffer\n");
        setCurrentStep(currentStep() + 1);
        mGradBuffer = std::move(res);
    } else {
        // current batch is not the first micro-batch
        // add grad to buffer
        printf("current batch is not the first micro-batch, add grad to buffer\n");
        for(auto& iter: res) {
            auto newVarp = mGradBuffer[iter.first] + iter.second;
            auto ptr = newVarp->readMap<float>();
            auto info = newVarp->getInfo();
            mGradBuffer[iter.first] = _Const(ptr, info->dim, info->order, info->type);;
        }
        if (mMicroStep % (mBatchSize / mMicroBatchSize) == 0) {
            //end current batch, update params by using gradBuffer
            printf("end current batch, update params by using gradBuffer\n");
            for(auto& iter: mGradBuffer) {
                iter.second = iter.second / _Scalar<float>(mBatchSize / mMicroBatchSize);
                iter.second.fix(Express::VARP::TRAINABLE);
            }
            for (auto iter : mGradBuffer) {
                iter.first->input(iter.second);
            }
        }
    }
    return true;
}

} // namespace Train
} // namespace MNN
