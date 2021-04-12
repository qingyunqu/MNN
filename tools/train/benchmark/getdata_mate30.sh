# adb push ~/Desktop/data/mnist_data /data/local/tmp
# adb push ~/Desktop/data/mobilenet_demo /data/local/tmp

# adb push ./usage_monitor.sh /data/local/tmp

# adb shell chmod 0777 /data/local/tmp/usage_monitor.sh

DeviceName=SamsungS8p_OPENCL
DEVICE=SamsungS8p_OPENCL
ANDROID_DIR=/data/local/tmp
OUT_FILE=train_bench

ModelName=Lenet
adb shell "cat /proc/cpuinfo > $ANDROID_DIR/$OUT_FILE.result"

function bench_session() {
    adb shell "mkdir -p $ANDROID_DIR/models"
    adb shell "mv $ANDROID_DIR/temp* $ANDROID_DIR/models"
    adb shell "cat /proc/cpuinfo > $ANDROID_DIR/bench_session.result"
    adb shell "echo "
    # echo "su; LD_LIBRARY_PATH=\$ANDROID_DIR; \$ANDROID_DIR/build/benchmark.out \$ANDROID_DIR/models/ 10 3 0 > \$ANDROID_DIR/bench_session.result" > ./cmd.txt # adb shell中连续执行多条指令的办法
    # adb shell < ./cmd.txt 
    adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/build/benchmark.out $ANDROID_DIR/models/ 10 3 0 >> $ANDROID_DIR/bench_session.result"
    adb pull $ANDROID_DIR/bench_session.result ./$DEVICE/
}

# ./bench_batchsize_unroot.sh $ModelName 1 -p
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_1.result
# ./bench_batchsize_unroot.sh $ModelName 2
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_2.result
# ./bench_batchsize_unroot.sh $ModelName 4 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_4.result
# ./bench_batchsize_unroot.sh $ModelName 8 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_8.result
# ./bench_batchsize_unroot.sh $ModelName 16 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_16.result
# # ./bench_batchsize_unroot.sh $ModelName 32 
# # adb pull $ANDROID_DIR/train_stamp.result ./
# # mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_32.result

# adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${ModelName}.result
# # adb shell < ./cmd.txt 
# adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${ModelName}.result
# sleep 200 # cooling device to avoid freq downgrade

ModelName=Alexnet
adb shell "cat /proc/cpuinfo > $ANDROID_DIR/$OUT_FILE.result"
./bench_batchsize_unroot.sh $ModelName 1
adb pull $ANDROID_DIR/train_stamp.result ./
mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_1.result
./bench_batchsize_unroot.sh $ModelName 2
adb pull $ANDROID_DIR/train_stamp.result ./
mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_2.result
./bench_batchsize_unroot.sh $ModelName 4 
adb pull $ANDROID_DIR/train_stamp.result ./
mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_4.result
./bench_batchsize_unroot.sh $ModelName 8 
adb pull $ANDROID_DIR/train_stamp.result ./
mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_8.result
# ./bench_batchsize_unroot.sh $ModelName 16 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_16.result
# ./bench_batchsize_unroot.sh $ModelName 32 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_32.result

adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${ModelName}.result
# adb shell < ./cmd.txt 
adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${ModelName}.result
# sleep 200 # cooling device to avoid freq downgrade

# ModelName=Squeezenet
# adb shell "cat /proc/cpuinfo > $ANDROID_DIR/$OUT_FILE.result"
# ./bench_batchsize_unroot.sh $ModelName 1
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_1.result
# ./bench_batchsize_unroot.sh $ModelName 2
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_2.result
# ./bench_batchsize_unroot.sh $ModelName 4 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_4.result
# ./bench_batchsize_unroot.sh $ModelName 8 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_8.result
# # ./bench_batchsize_unroot.sh $ModelName 16 
# # adb pull $ANDROID_DIR/train_stamp.result ./
# # mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_16.result
# # ./bench_batchsize_unroot.sh $ModelName 32 
# # adb pull $ANDROID_DIR/train_stamp.result ./
# # mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_32.result

# adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${ModelName}.result
# # adb shell < ./cmd.txt 
# adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${ModelName}.result
# sleep 200 # cooling device to avoid freq downgrade

# ModelName=GoogLenet
# adb shell "cat /proc/cpuinfo > $ANDROID_DIR/$OUT_FILE.result"
# ./bench_batchsize_unroot.sh $ModelName 1
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_1.result
# ./bench_batchsize_unroot.sh $ModelName 2
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_2.result
# ./bench_batchsize_unroot.sh $ModelName 4 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_4.result
# ./bench_batchsize_unroot.sh $ModelName 8 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_8.result
# # ./bench_batchsize_unroot.sh $ModelName 16 
# # adb pull $ANDROID_DIR/train_stamp.result ./
# # mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_16.result
# # ./bench_batchsize_unroot.sh $ModelName 32 
# # adb pull $ANDROID_DIR/train_stamp.result ./
# # mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_32.result

# adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${ModelName}.result
# bench_session
# # adb shell < ./cmd.txt 
# adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${ModelName}.result
# sleep 200 # cooling device to avoid freq downgrade

# ModelName=Mobilenet
# adb shell "cat /proc/cpuinfo > $ANDROID_DIR/$OUT_FILE.result"
# ./bench_batchsize_unroot.sh $ModelName 1
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_1.result
# ./bench_batchsize_unroot.sh $ModelName 2
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_2.result
# ./bench_batchsize_unroot.sh $ModelName 4 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_4.result
# ./bench_batchsize_unroot.sh $ModelName 8 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_8.result
# ./bench_batchsize_unroot.sh $ModelName 16 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_16.result
# ./bench_batchsize_unroot.sh $ModelName 32 
# adb pull $ANDROID_DIR/train_stamp.result ./
# mv -f ./train_stamp.result ./$DEVICE/train_stamp/$ModelName/train_stamp_32.result

# adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${ModelName}.result
# # adb shell < ./cmd.txt 
# adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${ModelName}.result

# sleep 200 # cooling device to avoid freq downgrade