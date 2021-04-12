# adb push ~/Desktop/data/mnist_data /data/local/tmp
# adb push ~/Desktop/data/mobilenet_demo /data/local/tmp

# adb push ./usage_monitor.sh /data/local/tmp

# adb shell chmod 0777 /data/local/tmp/usage_monitor.sh

DeviceName=Mate30_OPENCL

./bench_batchsize_unroot.sh Lenet 1 2 4 8 16 -p
adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_Lenet.result
# adb shell < ./cmd.txt 
adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_Lenet.result
sleep 200 # cooling device to avoid freq downgrade

./bench_batchsize_unroot.sh Alexnet 1 2 4 8 16
adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_Alexnet.result
# adb shell < ./cmd.txt 
adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_Alexnet.result
sleep 200 # cooling device to avoid freq downgrade

./bench_batchsize_unroot.sh GoogLenet 1 2 4 8 16
adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_GoogLenet.result
# adb shell < ./cmd.txt 
adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_GoogLenet.result
sleep 200 # cooling device to avoid freq downgrade 

./bench_batchsize_unroot.sh Squeezenet 1 2 4 8 16
adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_Squeezenet.result
# adb shell < ./cmd.txt 
adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_Squeezenet.result
sleep 200 # cooling device to avoid freq downgrade

./bench_batchsize_unroot.sh Mobilenet 1 2 4 8 16 32 





