set -e
ABI="arm64-v8a"
OPENCL="ON"
RUN_LOOP=10
FORWARD_TYPE=0
Para_Num=$#
PUSH_MODEL=${@: -1}
Para=$@
ModelName=$1
OUT_FILE=train_bench
DEVICE=XiaoMi9


WORK_DIR=`pwd`
BUILD_DIR=build
ANDROID_DIR=/data/local/tmp

function build_android_bench() {
    mkdir -p $BUILD_DIR
    mkdir -p $DEVICE/train_bench
    mkdir -p $DEVICE/train_stamp/$ModelName
    mkdir -p $DEVICE/usage_monitor
    mkdir -p $DEVICE/processed_data/$ModelName
    cd $BUILD_DIR
    cmake ../../../.. \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DANDROID_ABI="${ABI}" \
        -DANDROID_STL=c++_static \
        -DMNN_USE_LOGCAT=false \
        -DMNN_OPENCL:BOOL=$OPENCL \
        -DMNN_BUILD_BENCHMARK=ON \
        -DANDROID_NATIVE_API_LEVEL=android-21  \
        -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
        -DMNN_BUILD_TRAIN=ON \
        -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3

    make -j8 
}

function bench_android() {
    # start_monitor &
    # sleep 5
    build_android_bench
    find . -name "*.so" | while read solib; do
        adb push $solib  $ANDROID_DIR
    done
    
    if [ $PUSH_MODEL == "-p" ]; then
    echo Pushing models
    adb push ../build $ANDROID_DIR
    fi

    adb shell chmod 0777 $ANDROID_DIR/build/runTrainDemo.out
    
    # adb shell "cat /proc/cpuinfo > $ANDROID_DIR/$OUT_FILE.result"
    adb shell "echo "
    adb shell "loc=`pwd` echo $loc >> $ANDROID_DIR/$OUT_FILE.result" # 这里到底如何把安卓的pwd给写入到指定文件里呢
    adb shell "echo Build Flags: ABI=$ABI  OpenCL=$OPENCL >> $ANDROID_DIR/$OUT_FILE.result"
    
    for parameter in ${Para}
    do
        
        # when push model, ignore the last para
        if [ $parameter == "-p" ];then
        continue
        fi

        if [ $parameter == $ModelName ]; then
            echo "Choose $parameter"

            else
            echo "Testing $ModelName with BatchSize $parameter"

            # 如何进入到build文件夹执行runTrainDemo.out

            # benchmark  Alexnet or Lenet
            if [ $ModelName != Mobilenet ]; then
            adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/build/runTrainDemo.out MnistBenchmark $ANDROID_DIR/mnist_data $parameter $ModelName 2>$ANDROID_DIR/benchmark.err >> $ANDROID_DIR/$OUT_FILE.result"
            echo end
            fi

            # benchmark  MnistTrain
            if [ $ModelName == Mobilenet ]; then
            # echo "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/build/runTrainDemo.out MobilenetV2Train $ANDROID_DIR/mobilenet_demo/train_dataset/train_images/ $ANDROID_DIR/mobilenet_demo/train_dataset/train.txt $ANDROID_DIR/mobilenet_demo/test_dataset/test_images/ $ANDROID_DIR/mobilenet_demo/test_dataset/test.txt $BatchSize 2>$ANDROID_DIR/benchmark.err >> $ANDROID_DIR/$OUT_FILE"
            adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/build/runTrainDemo.out MobilenetV2Benchmark $ANDROID_DIR/mobilenet_demo/train_dataset/train_images/ $ANDROID_DIR/mobilenet_demo/train_dataset/train.txt $ANDROID_DIR/mobilenet_demo/test_dataset/test_images/ $ANDROID_DIR/mobilenet_demo/test_dataset/test.txt $parameter 2>$ANDROID_DIR/benchmark.err >> $ANDROID_DIR/$OUT_FILE.result"
            echo end
            fi

        fi
    done

}


bench_android
