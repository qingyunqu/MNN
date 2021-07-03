//
// Created by 王启鹏 on 2021/3/23.
//

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "DemoUnit.hpp"
#include "MobilenetV2.hpp"
#include "MobilenetV1.hpp"
#include "GoogLeNet.hpp"
#include "SqueezeNet.hpp"
#include "Alexnet.hpp"
#include "Lenet.hpp"
#include "MnistUtils.hpp"
#include "MobilenetV2Utils.hpp"
#include "MemTimeProfileUtils.cpp"
#include <MNN/expr/NN.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "module/PipelineModule.hpp"

using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class MemTimeProfile : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        RandomGenerator::generator(17);
        if (argc == 5) {
            std::string modelname = argv[1];
            if(modelname == "Lenet") {
                std::string root = argv[2];
                int batchsize = atoi(argv[3]);
                int microBatchsize = atoi(argv[4]);
                std::shared_ptr<Module> model(new Lenet);
                MnistUtils::train(model, root, batchsize, microBatchsize);
                return 0;
            }
        } else if (argc == 6) {
            std::string modelname = argv[1];
            if (modelname == "MobilenetV2" || modelname == "MobilenetV1" || modelname == "Alexnet" || modelname == "Squeezenet" || modelname == "Googlenet") {
                std::string trainImagesFolder = argv[2];
                std::string trainImagesTxt = argv[3];
                std::string testImagesFolder = argv[2];
                std::string testImagesTxt = argv[3];
                int batchsize = atoi(argv[4]);
                int microBatchsize = atoi(argv[5]);
                std::shared_ptr<Module> model;
                int numClass = 10;
                if (modelname == "MobilenetV2") {
                    numClass = 1001;
                    model = std::make_shared<MobilenetV2>();
                } else if (modelname == "MobilenetV1") {
                    numClass = 1001;
                    model = std::make_shared<MobilenetV1>();
                } else if (modelname == "Alexnet") {
                    model = std::make_shared<Alexnet>();
                } else if (modelname == "Squeezenet") {
                    model = std::make_shared<Squeezenet>();
                } else if (modelname == "Googlenet") {
                    model = std::make_shared<GoogLenet>();
                }
                MemTimeProfileUtils::train(model, numClass, 1, trainImagesFolder, trainImagesTxt, testImagesFolder, testImagesTxt, batchsize, microBatchsize);
                return 0;
            }
        }
        std::cout << "usage: \n"
                     "\t./runTrainDemo.out MemTimeProfile Lenet /path/to/unzipped/mnist/data/ BatchSize MicroBatchSize\n"
                     "\t./runTrainDemo.out MemTimeProfile MODEL path/to/images/ path/to/txt BatchSize MicroBatchSize\n"
                     "\t\tMODEL=[MobilenetV2|Alexnet|Squeezenet|Googlenet]\n";
        return 0;

    }
};


DemoUnitSetRegister(MemTimeProfile, "MemTimeProfile");