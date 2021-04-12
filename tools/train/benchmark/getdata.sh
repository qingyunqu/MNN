NetNames=("Lenet" "GoogLenet" "Squeezenet" "Alexnet" "Mobilenet")
# NetNames=("Mobilenet")
DeviceName="MEIZU_big"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./bench_batchsize_root.sh ${NetName} 1 2 4 8 16 32 -p
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 300 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done





# ./bench_batchsize_android.sh Squeezenet 1 2 4 8 16
# adb pull /data/local/tmp/train_bench.result ./MEIZU_OPENCL/train_bench/train_bench_Squeezenet.result
# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法
# adb shell < ./cmd.txt 
# adb pull /data/local/tmp/usage_monitor.result ./MEIZU_OPENCL/usage_monitor/usage_monitor_Squeezenet.result
# sleep 200

# ./bench_batchsize_android.sh Alexnet 1 2 4 8
# adb pull /data/local/tmp/train_bench.result ./MEIZU_OPENCL/train_bench/train_bench_Alexnet.result
# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法
# adb shell < ./cmd.txt 
# adb pull /data/local/tmp/usage_monitor.result ./MEIZU_OPENCL/usage_monitor/usage_monitor_Alexnet.result
# sleep 200

# ./bench_batchsize_android.sh Lenet 1 2 4 8 
# adb pull /data/local/tmp/train_bench.result ./MEIZU_OPENCL/train_bench/train_bench_Lenet.result
# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法
# adb shell < ./cmd.txt 
# adb pull /data/local/tmp/usage_monitor.result ./MEIZU_OPENCL/usage_monitor/usage_monitor_Lenet.result
# #./bench_batchsize_android.sh Lenet 4 

# sleep 200
# ./bench_batchsize_android.sh Mobilenet 1 2 4 8 16 32
# sleep 200


# #./bench_batchsize_android.sh Lenet 1 -p
# #./bench_batchsize_android.sh Lenet 4 
# #sleep 200
# ./bench_batchsize_android.sh GoogLenet 32 
# #sleep 200
# ./bench_batchsize_android.sh Squeezenet 32 
# #sleep 200
# ./bench_batchsize_android.sh Alexnet 32 
# #sleep 200
# ./bench_batchsize_android.sh Mobilenet 32 
# #sleep 200
