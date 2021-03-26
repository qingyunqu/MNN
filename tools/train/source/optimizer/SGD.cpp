//
//  SGD.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SGD.hpp"
#include "OpGrad.hpp"
using namespace MNN::Express;

namespace MNN {
namespace Train {
SGD::SGD(std::shared_ptr<Module> module) : ParameterOptimizer(module) {
    auto train = ParameterOptimizer::trainable();
    for (auto p : train) {
        mHistory[p] = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
    }
}

void SGD::setLearningRate(float rate) {
    mLearningRate = rate;
}

void SGD::setMomentum(float momentum) {
    mMomentum = momentum;
}

void SGD::setWeightDecay(float decay) {
    mWeightDecay = decay;
}

void SGD::setRegularizationMethod(RegularizationMethod method) {
    mRegularizationMethod = method;
}

float SGD::currentLearningRate() {
    return mLearningRate;
}

float SGD::getMomentum() {
    return mMomentum;
}

float SGD::getWeightDecay() {
    return mWeightDecay;
}

SGD::RegularizationMethod SGD::getRegularizationMethod() {
    return mRegularizationMethod;
}

Express::VARP SGD::regularizeParameters(Express::VARP param, Express::VARP grad) {
    VARP addWeightDecayGrad;
    if (mRegularizationMethod == L1) {
        auto temp          = _Sign(param);
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * temp + grad;
    } else if (mRegularizationMethod == L2) {
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * param + grad;
    } else if (mRegularizationMethod == L1L2) {
        auto temp          = _Sign(param);
        auto L1 = _Const(mWeightDecay, {}, NCHW) * temp;
        auto L2 = _Const(mWeightDecay, {}, NCHW) * param;
        addWeightDecayGrad = L1 + L2 + grad;
    }

    return addWeightDecayGrad;
}

Express::VARP SGD::onComputeUpdateValue(Express::VARP param, Express::VARP grad) {
    auto lr         = _Const(mLearningRate, {}, NCHW);
    mHistory[param] = lr * grad + _Const(mMomentum, {}, NCHW) * mHistory[param];
    mHistory[param].fix(Express::VARP::CONSTANT);
    //FUNC_PRINT_ALL(_ReduceMax(grad)->readMap<float>()[0], f);
    return mHistory[param];
}

std::map<Express::VARP, Express::VARP> SGD::onGetNextParameter(Express::VARP loss) {
    printf("begin compute grad graph\n");
    auto grad = OpGrad::grad(loss, trainable(), mGradBlockExprName);
    printf("finish compute grad graph\n");
    auto parameters = module()->parameters();
    std::vector<VARP> prepareCompute;
    for (auto iter : parameters) {
        if (iter->expr().first->get() != nullptr) {
            prepareCompute.emplace_back(iter);
        }
    }
    for (auto& iter : grad) {
        prepareCompute.emplace_back(iter.second);
    }
    printf("begin prepare compute\n");
    Variable::prepareCompute(prepareCompute);
    printf("finish prepare compute & start read-map\n");
    std::vector<VARP> replaceOp(prepareCompute.size());
    for (int i=0; i<prepareCompute.size(); ++i) {
        auto info = prepareCompute[i]->getInfo();
        auto ptr = prepareCompute[i]->readMap<void>();
        if (nullptr == ptr) {
            MNN_ERROR("Compute error in SGD\n");
            return {};
        }
        auto newVar = _Const(ptr, info->dim, info->order, info->type);
        replaceOp[i]= newVar;
    }
    printf("finish read-map & start replace\n");
    for (int i=0; i<prepareCompute.size(); ++i) {
        Variable::replace(prepareCompute[i], replaceOp[i]);
    }
    printf("finish replace & start compute update value\n");
    for (auto& iter : grad) {
        // apply regularization
        MNN_MEMORY_PROFILE("\t")
        auto addWeightDecayGrad = regularizeParameters(iter.first, iter.second);
        addWeightDecayGrad.fix(Express::VARP::CONSTANT);
        // apply momentum, etc.
        auto updateValue = this->onComputeUpdateValue(iter.first, addWeightDecayGrad);
        // apply update
        auto newParameter = iter.first - updateValue;
        iter.second       = newParameter;
    }
    printf("finish func %s\n", __FUNCTION__ );
    return grad;
}

} // namespace Train
} // namespace MNN
