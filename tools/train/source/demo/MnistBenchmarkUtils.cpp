//
//  MnistBenchmarkUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MnistBenchmarkUtils.hpp"
#include <MNN/expr/Executor.hpp>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include "NormalDataset.hpp"
#include "MnistDataset.hpp"
#include <MNN/expr/NN.hpp>
#include "SGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <sys/time.h>
#include "ADAM.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "OpGrad.hpp"
using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

void MnistBenchmarkUtils::train(std::shared_ptr<Module> model, std::string root, int BatchSize, std::string NetName) {
    {
        // 更换模型之后如果还调用之前的.mnn则会报错，invalid net
        // Load snapshot
        auto para = Variable::load("mnist.snapshot.mnn");
        
        model->loadParameters(para);
        
    }
    system("echo Begin testing : $(date +%s%3N) > /data/local/tmp/train_stamp.result");
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 4);
    std::shared_ptr<SGD> sgd(new SGD(model));
    // FUNC_PRINT("happy");
    sgd->setMomentum(0.9f);
    // sgd->setMomentum2(0.99f);
    sgd->setWeightDecay(0.0005f);

    auto dataset = NetName == "Lenet" ? MnistDataset::create(root, MnistDataset::Mode::TRAIN) : NormalDataset::create(root, NormalDataset::Mode::TRAIN);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    const size_t batchSize  = BatchSize;
    // MNN_PRINT("%s | %d | %s() : BatchSize %d \n", strrchr(__FILE__, '/')+1, __LINE__, __FUNCTION__, batchSize);
    const size_t numWorkers = 0;
    bool shuffle            = true;
    system("echo Load data : $(date +%s%3N) >> /data/local/tmp/train_stamp.result");
    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

    size_t iterations = dataLoader->iterNumber();

    auto testDataset            = NetName == "Lenet" ? MnistDataset::create(root, MnistDataset::Mode::TEST) : NormalDataset::create(root, NormalDataset::Mode::TEST);
    // auto testDataset            = NormalDataset::create(root, NormalDataset::Mode::TEST);
    const size_t testBatchSize  = BatchSize;
    const size_t testNumWorkers = 0;
    shuffle                     = false;

    auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));

    size_t testIterations = testDataLoader->iterNumber();

    // set measurement parameter
    int warmUp = 5;
    int measureIterations = 128/BatchSize + 10;
    // int loops = 1;
    if(NetName == "Lenet") { measureIterations = 128/BatchSize + 200; } 
    // if(NetName == "Lenet") { loops = 1; } // Lenet train too fast, loop it for more samples !!! loop will lose precision

    MNN_PRINT(" warmUp: %d \n measureIteration: %d \n", warmUp, measureIterations);
    
    std::vector<float> latency;
    for (int epoch = 0; epoch < 1; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            // AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);
            Timer _100Time;
            int lastIndex = 0;
            int moveBatchSize = 0;
            system("echo Begin training : $(date +%s%3N) >> /data/local/tmp/train_stamp.result");
            for (int i = 0; i < warmUp + measureIterations +1; i++) {
                // MNN_PRINT("i %d\n", i);
            // for (int i = 0; i < iterations; i++) {
                // AUTOTIME;
                // if(i == iterations) { measureIterations = iterations - warmUp - 1; break; }
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto cast       = _Cast<float>(example.first[0]);
                example.first[0] = cast * _Const(1.0f / 255.0f);
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(10), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));

                auto predict = model->forward(example.first[0]);
                auto loss    = _CrossEntropy(predict, newTarget);
//#define DEBUG_GRAD
#ifdef DEBUG_GRAD
                {
                    static bool init = false;
                    if (!init) {
                        init = true;
                        std::set<VARP> para;
                        example.first[0].fix(VARP::INPUT);
                        newTarget.fix(VARP::CONSTANT);
                        auto total = model->parameters();
                        for (auto p :total) {
                            para.insert(p);
                        }
                        auto grad = OpGrad::grad(loss, para);
                        total.clear();
                        for (auto iter : grad) {
                            total.emplace_back(iter.second);
                        }
                        Variable::save(total, ".temp.grad");
                    }
                }
#endif
                float rate   = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);
                if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
                    std::cout << "epoch: " << (epoch);
                    std::cout << "  " << moveBatchSize << " / " << dataLoader->size();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate;
                    std::cout << " time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << (i - lastIndex) <<  " iter"  << std::endl;
                    std::cout.flush();
                    _100Time.reset();
                    lastIndex = i;
                }
                {
                struct timeval Current;
                gettimeofday(&Current, nullptr);
                uint64_t mLastResetTime = Current.tv_sec * 1000000 + Current.tv_usec;

                    sgd->step(loss);                

                gettimeofday(&Current, nullptr);
                auto lastTime = Current.tv_sec * 1000000 + Current.tv_usec;
                auto durations = lastTime - mLastResetTime;
                
                // MNN_PRINT("duration is %f ms \n", (float)durations / 1000.0f);
                latency.push_back((float)durations / 1000.0f);
                // MNN_PRINT("第%d个iteration的latency是：%f\n", i, latency.back());
                }
            }
            // 去掉前3个warmup，从第四个latency开始计算，算20个latency的均值
            float latency_count = 0;
            for (int i = warmUp; i < warmUp + measureIterations; i++){
                latency_count += latency[i];
            }
            auto latency_avg = latency_count / float(measureIterations);
            MNN_PRINT("Training latency of %s is : %f ms (batchsize is %d)\n", NetName.c_str(), latency_avg, BatchSize);
        }
        system("echo End trainning : $(date +%s%3N) >> /data/local/tmp/train_stamp.result");

        Variable::save(model->parameters(), "mnist.snapshot.mnn");
        {
            model->setIsTraining(false);
            auto forwardInput = NetName == "Lenet" ? _Input({BatchSize, 1, 28, 28}, NC4HW4) : _Input({BatchSize, 3, 224, 224}, NC4HW4);
            forwardInput->setName("data");
            auto predict = model->forward(forwardInput);
            predict->setName("prob");
            Transformer::turnModelToInfer()->onExecute({predict});
            std::string modelName = "/data/local/tmp/temp." + NetName + "_" + std::to_string(BatchSize) + ".mnn";
            MNN_PRINT("modeName is %s\n", modelName.c_str());
            Variable::save({predict}, modelName.c_str());
        }

        latency.clear();
        // measureIterations = measureIterations/10;
        measureIterations = testIterations - warmUp - 10;
        int correct = 0;
        testDataLoader->reset();
        model->setIsTraining(false);
        int moveBatchSize = 0;
        system("echo Begin inferring : $(date +%s%3N) >> /data/local/tmp/train_stamp.result");
        //skip infer for quick temper rise
        // for (int i = 0; i < 0; i++) { 
        for (int i = 0; i < warmUp + measureIterations +1; i++) {
            // if(i >= testIterations) { break; }
            auto data       = testDataLoader->next();
            auto example    = data[0];
            moveBatchSize += example.first[0]->getInfo()->dim[0];
            if ((i + 1) % 100 == 0) {
                std::cout << "test: " << moveBatchSize << " / " << testDataLoader->size() << std::endl;
            }
            auto cast       = _Cast<float>(example.first[0]);
            example.first[0] = cast * _Const(1.0f / 255.0f);
            MNN:Express:VARP predict;
            
            
            predict    = model->forward(example.first[0]);
            

            // opencl and inputSize=224 hasn't support ArgMax op yet
            // predict         = _ArgMax(predict, 1);
            // auto accu       = _Cast<int32_t>(_Equal(predict, _Cast<int32_t>(example.second[0]))).sum({});
            {   
            struct timeval Current;
            gettimeofday(&Current, nullptr);
            uint64_t mLastResetTime = Current.tv_sec * 1000000 + Current.tv_usec;
            
            correct += predict->readMap<int32_t>()[0];

            gettimeofday(&Current, nullptr);
            auto lastTime = Current.tv_sec * 1000000 + Current.tv_usec;
            auto durations = lastTime - mLastResetTime;
            latency.push_back((float)durations / 1000.0f);
            }
        }
        system("echo End inferring : $(date +%s%3N) >> /data/local/tmp/train_stamp.result");
        float latency_count = 0;
        for (int i = warmUp; i < warmUp + measureIterations; i++){
                latency_count += latency[i];
            }
        auto latency_avg = latency_count / float(measureIterations);
        MNN_PRINT("Inferring latency of %s is : %f ms (batchsize is %d)\n", NetName.c_str(), latency_avg, BatchSize);
       
        auto accu = (float)correct / (float)testDataLoader->size();
        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;
        exe->dumpProfile();
    }
}
