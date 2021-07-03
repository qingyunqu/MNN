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
    MNN_PRINT("%s:%s: begin compute grad graph\n", __FILE_NAME__, __FUNCTION__ );
    auto grad = OpGrad::grad(loss, trainable(), mGradBlockExprName);
    MNN_PRINT("%s:%s: finish compute grad graph\n", __FILE_NAME__, __FUNCTION__ );
    auto parameters = module()->parameters();
    std::vector<VARP> prepareCompute;
//    int tot_size = 0;
//    for (auto var: trainable()) {
//        tot_size += var->getInfo()->size;
//    }
//    printf("trainable.size = %d\n", tot_size);
//
    auto execOrder = Variable::getExecuteOrder({loss});
    std::map<EXPRP, VARP> paramExpr;
    for(auto param: parameters){
        paramExpr.insert(std::make_pair(param->expr().first, param));
    }
    for (auto iter = execOrder.rbegin(); iter != execOrder.rend(); iter++) {
        if (paramExpr.find(*iter) != paramExpr.end()) {
            prepareCompute.emplace_back(grad[paramExpr[*iter]]);
        }
    }
//    int untrainablesize = 0;
//    for (auto iter : parameters) {
//        if (iter->expr().first->get() != nullptr) {
//            untrainablesize += iter->getInfo()->size;
//            prepareCompute.emplace_back(iter);
//        }
//    }
//    printf("untrainable.size = %d\n", untrainablesize);
//    for (auto& iter : grad) {
//        prepareCompute.emplace_back(iter.second);
//    }
    MNN_PRINT("%s:%s: prepareCompute.size() = %d\ttrainable.size() = %d\n",  __FILE_NAME__, __FUNCTION__ , prepareCompute.size(), trainable().size());
    MNN_PRINT("%s:%s: begin enableHeuristicAlloc\n", __FILE_NAME__, __FUNCTION__ );
//    Variable::enableHeuristicAlloc(true);
    MNN_PRINT("%s:%s: begin prepare compute\n", __FILE_NAME__, __FUNCTION__ );
    Variable::prepareCompute(prepareCompute);
    MNN_PRINT("%s:%s: finish prepare compute & start read-map\n", __FILE_NAME__, __FUNCTION__ );
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
    Variable::enableHeuristicAlloc(false);
    MNN_PRINT("%s:%s: finish read-map & start replace\n", __FILE_NAME__, __FUNCTION__ );
    for (int i=0; i<prepareCompute.size(); ++i) {
//        MNN_ASSERT(prepareCompute[i]->expr().first->outputSize() == replaceOp[i]->expr().first->outputSize())
        Variable::replace(prepareCompute[i], replaceOp[i]);
//        MNN_PRINT("finish replace var[%d]\n", i)
    }
    MNN_PRINT("%s:%s: finish replace & start compute update value\n", __FILE_NAME__, __FUNCTION__ );
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
    MNN_PRINT("%s: finish func %s\n", __FILE_NAME__ , __FUNCTION__ );
    return grad;
}

void SGD::profile(Express::VARP loss) {
    printf("call %s\n", __FUNCTION__ );
    auto grad = OpGrad::grad(loss, trainable(), mGradBlockExprName);
    printf("finish opgrad in %s\n", __FUNCTION__ );
    auto parameters = module()->parameters();
    std::vector<VARP> prepareCompute;
    auto execOrder = Variable::getExecuteOrder({loss});
    std::map<EXPRP, VARP> paramExpr;
    for(auto param: parameters){
        paramExpr.insert(std::make_pair(param->expr().first, param));
    }
    int insertPos = 0;
    for(auto expr: execOrder) {
        if (paramExpr.find(expr)!=paramExpr.end()) {
            if (expr->get() != nullptr) {
                prepareCompute.insert(prepareCompute.begin()+insertPos, paramExpr[expr]);
                insertPos++;
            } else {
                prepareCompute.insert(prepareCompute.begin()+insertPos, grad[paramExpr[expr]]);
            }
        }
    }
//    for (auto iter : parameters) {
//        if (iter->expr().first->get() != nullptr) {
//            prepareCompute.emplace_back(iter);
//        }
//    }
//    for (auto& iter : grad) {
//        prepareCompute.emplace_back(iter.second);
//    }
    Variable::prepareCompute(prepareCompute);
    printf("finish prepareCompute in %s\n", __FUNCTION__ );
    Variable::profileExecute(prepareCompute[0]);
//    Variable::clearCache(prepareCompute);
}

} // namespace Train
} // namespace MNN
