//
//  MobilenetV2Utils.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MemTimeProfileUtils_CPP
#define MemTimeProfileUtils_CPP

#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <string>
#include <cmath>
#include <iostream>
#include <vector>
#include "DemoUnit.hpp"
#include "MicroSGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "Loss.hpp"
#include "Transformer.hpp"
#include "ImageDataset.hpp"
#include "module/PipelineModule.hpp"
#include "OpGrad.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

class MemTimeProfileUtils {
public:
    static void train(std::shared_ptr<MNN::Express::Module> model, const int numClasses, const int addToLabel,
                      std::string trainImagesFolder, std::string trainImagesTxt,
                      std::string testImagesFolder, std::string testImagesTxt,
                      const int batchsize = -1, const int microBatchsize = -1,
                      const int trainQuantDelayEpoch = 10, const int quantBits = 8) {
//        auto modelMap1 = Express::Variable::load("mnist.snapshot1.mnn");
//        auto modelMap2 = Express::Variable::load("mnist.snapshot1.mnn");
//        auto modelMap3 = Express::Variable::load("mnist.snapshot1.mnn");
//        for (int i = 0; i < modelMap1.size(); ++i) {
//            modelMap3[i]->input((modelMap1[i] + modelMap2[i]) / _Scalar<float>(2));
//        }
//        Express::Variable::save(modelMap3, "mnist.snapshot.agg.mnn");
//        return;

        auto exe = Executor::getGlobalExecutor();
        BackendConfig config;
        exe->setGlobalExecutorConfig(MNN_FORWARD_USER_1, config, 2);
//    std::shared_ptr<SGD> solver(new SGD(model));

        int trainBatchSize, trainMicroBatchsize;
        int trainNumWorkers = 1;
        int testBatchSize = 1;
        int testNumWorkers = 0;
        if (batchsize != -1 && microBatchsize != -1 && batchsize % microBatchsize == 0) {
            trainBatchSize = batchsize;
            trainMicroBatchsize = microBatchsize;
        } else {
            trainBatchSize = trainMicroBatchsize = 8;
        }
        exe->configHeuristicStrategy(model->name(), trainMicroBatchsize);

        std::shared_ptr<MicroSGD> solver(new MicroSGD(model, trainBatchSize, trainMicroBatchsize));
        solver->setMomentum(0.9f);
        // solver->setMomentum2(0.99f);
        solver->setWeightDecay(0.00004f);

        auto converImagesToFormat  = CV::RGB;
        int resizeHeight           = 224;
        int resizeWidth            = 224;
        std::vector<float> means = {127.5, 127.5, 127.5};
        std::vector<float> scales = {1/127.5, 1/127.5, 1/127.5};
        std::vector<float> cropFraction = {0.875, 0.875}; // center crop fraction for height and width
        bool centerOrRandomCrop = false; // true for random crop
        std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(
                ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight, resizeWidth, scales, means,cropFraction, centerOrRandomCrop));
        bool readAllImagesToMemory = false;
        auto trainDataset = ImageDataset::create(trainImagesFolder, trainImagesTxt, datasetConfig.get(), readAllImagesToMemory);
        auto testDataset = ImageDataset::create(testImagesFolder, testImagesTxt, datasetConfig.get(), readAllImagesToMemory);

        auto trainDataLoader = trainDataset.createLoader(trainMicroBatchsize, true, true, trainNumWorkers);
        auto testDataLoader = testDataset.createLoader(testBatchSize, true, true, testNumWorkers);

        const int trainIterations = trainDataLoader->iterNumber();
        const int testIterations = testDataLoader->iterNumber();




        // const int usedSize = 1000;
        // const int testIterations = usedSize / testBatchSize;
        if (0){
            // measure phase
            printf("measuremene phase\n");
            model->clearCache();
            exe->gc(Executor::FULL);
            exe->resetProfile();
            trainDataLoader->reset();
            model->setIsTraining(true);
            auto trainData  = trainDataLoader->next();
            auto example    = trainData[0];

            // Compute One-Hot
            auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(example.second[0] + _Scalar<int32_t>(addToLabel), {1})),
                                     _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                                     _Scalar<float>(0.0f));

            auto predict = model->forward(_Convert(example.first[0], NC4HW4));
            auto loss    = _CrossEntropy(predict, newTarget);
            solver->setLearningRate(1e-5);
            printf("start profile\n");
            solver->profile(loss);
            printf("finish profile execution\n");
//            MNN_ASSERT(0)
        }

        for (int epoch = 0; epoch < 50; ++epoch) {
            model->clearCache();
            exe->gc(Executor::FULL);
            exe->resetProfile();
            {
                AUTOTIME;
                trainDataLoader->reset();
                model->setIsTraining(true);
                // turn float model to quantize-aware-training model after a delay
                if (epoch == trainQuantDelayEpoch) {
                    // turn model to train quant model
                    std::static_pointer_cast<PipelineModule>(model)->toTrainQuant(quantBits);
                }
                for (int i = 0; i < 5; i++) {
                    if (i % trainIterations == 0) {
                        trainDataLoader->reset();
                    }
                    AUTOTIME;
                    MNN_MEMORY_PROFILE("begin an iteration")
                    MNN_PRINT("begin an iteration %d\n", i);
                    auto trainData  = trainDataLoader->next();
                    auto example    = trainData[0];

                    // Compute One-Hot
                    auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(example.second[0] + _Scalar<int32_t>(addToLabel), {1})),
                                             _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                                             _Scalar<float>(0.0f));

                    auto predict = model->forward(_Convert(example.first[0], NC4HW4));
                    auto loss    = _CrossEntropy(predict, newTarget);
                    // float rate   = LrScheduler::inv(0.0001, solver->currentStep(), 0.0001, 0.75);
                    float rate = 1e-5;
//                    loss->readMap<float>();
//                    return;
                    solver->setLearningRate(rate);
//                if (solver->currentStep() % 10 == 0) {
//                    std::cout << "train iteration: " << solver->currentStep();
//                    std::cout << " loss: " << loss->readMap<float>()[0];
//                    std::cout << " lr: " << rate << std::endl;
//                }
                    solver->step(loss);
//                    loss->readMap<float>();
//                    return;
                }
                return;
            }

            int correct = 0;
            int sampleCount = 0;
            testDataLoader->reset();
            model->setIsTraining(false);
            exe->gc(Executor::PART);

            AUTOTIME;
            for (int i = 0; i < testIterations; i++) {
                auto data       = testDataLoader->next();
                auto example    = data[0];
                auto predict    = model->forward(_Convert(example.first[0], NC4HW4));
                predict         = _ArgMax(predict, 1); // (N, numClasses) --> (N)
                auto label = _Squeeze(example.second[0]) + _Scalar<int32_t>(addToLabel);
                sampleCount += label->getInfo()->size;
                auto accu       = _Cast<int32_t>(_Equal(predict, label).sum({}));
                correct += accu->readMap<int32_t>()[0];

                if ((i + 1) % 10 == 0) {
                    std::cout << "test iteration: " << (i + 1) << " ";
                    std::cout << "acc: " << correct << "/" << sampleCount << " = " << float(correct) / sampleCount * 100 << "%";
                    std::cout << std::endl;
                }
            }
            auto accu = (float)correct / testDataLoader->size();
            // auto accu = (float)correct / usedSize;
            std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;

            {
                auto forwardInput = _Input({1, 3, resizeHeight, resizeWidth}, NC4HW4);
                forwardInput->setName("data");
                auto predict = model->forward(forwardInput);
                Transformer::turnModelToInfer()->onExecute({predict});
                predict->setName("prob");
                Variable::save({predict}, "temp.mobilenetv2.mnn");
            }

            exe->dumpProfile();
        }
    }
};

#endif