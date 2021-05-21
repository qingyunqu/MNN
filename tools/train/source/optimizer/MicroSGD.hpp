//
//  MicroSGD.hpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MicroSGD_hpp
#define MicroSGD_hpp

#include <MNN/expr/ExprCreator.hpp>
#include <string>
#include <vector>
#include "SGD.hpp"
#include "ParameterOptimizer.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

namespace MNN {
namespace Train {

class MNN_PUBLIC MicroSGD : public SGD {
public:
    MicroSGD(std::shared_ptr<Express::Module> module, int batchSize=1, int microBatchSize=1);
    virtual ~ MicroSGD() = default;
    bool step(Express::VARP loss);

protected:
    float mLearningRate                        = 0.001f;
    float mMomentum                            = 0;
    float mWeightDecay                         = 0;
    RegularizationMethod mRegularizationMethod = L2;
    std::map<MNN::Express::VARP, MNN::Express::VARP> mHistory;

    // For Cache
    const Express::Expr* mLoss = nullptr;
    int mLossFromIndex         = 0;
    std::string mGradBlockExprName;
    std::map<Express::VARP, Express::VARP> mGradBuffer;
    int mMicroStep = 0;
    int mBatchSize = 1;
    int mMicroBatchSize = 1;
};

} // namespace Train
} // namespace MNN

#endif // MicroSGD_hpp
