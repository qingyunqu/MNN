NetNames=( "Mobilenet" "GoogLenet" "Lenet" "Squeezenet" "Alexnet")

# numCore = 1 2 4 6 8 3 5 7
# change the numCore 

# cat ./mnist_1.txt > ../source/demo/MnistBenchmarkUtils.cpp
# cat ./mobile_1.txt > ../source/demo/mobilenetV2BenchmarkUtils.cpp

# DeviceName="MEIZU_numCore_1"

# adb shell echo "1 >> /data/local/tmp/control.txt"

# for NetName in ${NetNames[@]}
# do
#     ./bench_batchsize_root.sh ${NetName} 1 2 4 8 16 32 _1
#     # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
#     # adb shell < ./cmd.txt 
#     adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
#     sleep 30 # cooling device to avoid freq downgrade
#     # python ./dataprocess.py ${DeviceName} ${NetName}
# done

# sleep 200


cat ./mnist_2.txt > ../source/demo/MnistBenchmarkUtils.cpp
cat ./mobile_2.txt > ../source/demo/mobilenetV2BenchmarkUtils.cpp

DeviceName="MEIZU_numCore_2"

adb shell echo "2 >> /data/local/tmp/control.txt"

for NetName in ${NetNames[@]}
do
    ./bench_batchsize_root.sh ${NetName} 1 2 4 8 16 32 _2
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

sleep 200

cat ./mnist_4.txt > ../source/demo/MnistBenchmarkUtils.cpp
cat ./mobile_4.txt > ../source/demo/mobilenetV2BenchmarkUtils.cpp

DeviceName="MEIZU_numCore_4"

adb shell echo "4 >> /data/local/tmp/control.txt"

for NetName in ${NetNames[@]}
do
    ./bench_batchsize_root.sh ${NetName} 1 2 4 8 16 32 _4
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

sleep 200

cat ./mnist_6.txt > ../source/demo/MnistBenchmarkUtils.cpp
cat ./mobile_6.txt > ../source/demo/mobilenetV2BenchmarkUtils.cpp

DeviceName="MEIZU_numCore_6"

adb shell echo "6 >> /data/local/tmp/control.txt"

for NetName in ${NetNames[@]}
do
    ./bench_batchsize_root.sh ${NetName} 1 2 4 8 16 32 _6
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

sleep 200

cat ./mnist_8.txt > ../source/demo/MnistBenchmarkUtils.cpp
cat ./mobile_8.txt > ../source/demo/mobilenetV2BenchmarkUtils.cpp

DeviceName="MEIZU_numCore_8"

adb shell echo "8 >> /data/local/tmp/control.txt"

for NetName in ${NetNames[@]}
do
    ./bench_batchsize_root.sh ${NetName} 1 2 4 8 16 32 _8
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

sleep 200

cat ./mnist_3.txt > ../source/demo/MnistBenchmarkUtils.cpp
cat ./mobile_3.txt > ../source/demo/mobilenetV2BenchmarkUtils.cpp

DeviceName="MEIZU_numCore_3"

adb shell echo "3 >> /data/local/tmp/control.txt"

for NetName in ${NetNames[@]}
do
    ./bench_batchsize_root.sh ${NetName} 1 2 4 8 16 32 _3
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

sleep 200

cat ./mnist_5.txt > ../source/demo/MnistBenchmarkUtils.cpp
cat ./mobile_5.txt > ../source/demo/mobilenetV2BenchmarkUtils.cpp

DeviceName="MEIZU_numCore_5"

adb shell echo "5 >> /data/local/tmp/control.txt"

for NetName in ${NetNames[@]}
do
    ./bench_batchsize_root.sh ${NetName} 1 2 4 8 16 32 _5
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

sleep 200

cat ./mnist_7.txt > ../source/demo/MnistBenchmarkUtils.cpp
cat ./mobile_7.txt > ../source/demo/mobilenetV2BenchmarkUtils.cpp

DeviceName="MEIZU_numCore_7"

adb shell echo "7 >> /data/local/tmp/control.txt"

for NetName in ${NetNames[@]}
do
    ./bench_batchsize_root.sh ${NetName} 1 2 4 8 16 32 _7
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

sleep 200