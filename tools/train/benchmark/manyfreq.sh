NetNames=("GoogLenet" "Lenet" "Squeezenet" "Alexnet" "Mobilenet")


# adb shell "echo 1804800 > /data/local/tmp/control.txt"

# DeviceName="RedmiNote9P_1804800"

# # echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法
# for NetName in ${NetNames[@]}
# do
#     ./bench_batchsize_root.sh ${NetName} 1 2 4 8 16 32 -p 
#     # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
#     # adb shell < ./cmd.txt 
#     adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
#     sleep 30 # cooling device to avoid freq downgrade
#     # python ./dataprocess.py ${DeviceName} ${NetName}
# done

# sleep 200

# adb shell "echo 1708800 > /data/local/tmp/control.txt"

# DeviceName="RedmiNote9P_1708800"

# # echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法
# for NetName in ${NetNames[@]}
# do
#     ./a1708800.sh ${NetName} 1 2 4 8 16 32 -p 
#     # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
#     # adb shell < ./cmd.txt 
#     adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
#     sleep 30 # cooling device to avoid freq downgrade
#     # python ./dataprocess.py ${DeviceName} ${NetName}
# done

# sleep 200

# adb shell "echo 1612800 >> /data/local/tmp/control.txt"

# DeviceName="RedmiNote9P_1612800"

# # echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

# for NetName in ${NetNames[@]}
# do
#     ./a1612800.sh ${NetName} 1 2 4 8 16 32 -p
#     # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
#     # adb shell < ./cmd.txt 
#     adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
#     sleep 30 # cooling device to avoid freq downgrade
#     # python ./dataprocess.py ${DeviceName} ${NetName}
# done

# sleep 200

# adb shell "echo 1516800 >> /data/local/tmp/control.txt"

# DeviceName="RedmiNote9P_1516800"

# # echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

# for NetName in ${NetNames[@]}
# do
#     ./a1516800.sh ${NetName} 1 2 4 8 16 32 -p
#     # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
#     # adb shell < ./cmd.txt 
#     adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
#     sleep 30 # cooling device to avoid freq downgrade
#     # python ./dataprocess.py ${DeviceName} ${NetName}
# done

# sleep 200

# adb shell "echo 1324800 >> /data/local/tmp/control.txt"

# DeviceName="RedmiNote9P_1324800"

# # echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

# for NetName in ${NetNames[@]}
# do
#     ./a1324800.sh ${NetName} 1 2 4 8 16 32 -p
#     # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
#     # adb shell < ./cmd.txt 
#     adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
#     sleep 30 # cooling device to avoid freq downgrade
#     # python ./dataprocess.py ${DeviceName} ${NetName}
# done

# sleep 200

# adb shell echo "1017600 >> /data/local/tmp/control.txt"

# DeviceName="RedmiNote9P_1017600"

# # echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

# for NetName in ${NetNames[@]}
# do
#     ./a1017600.sh ${NetName} 1 2 4 8 16 32 -p
#     # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
#     # adb shell < ./cmd.txt 
#     adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
#     sleep 200 # cooling device to avoid freq downgrade
#     # python ./dataprocess.py ${DeviceName} ${NetName}
# done

# sleep 200



big kernel

adb shell echo "787200 >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_big_787200"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./benchmark_batchsize_freq.sh ${NetName} 1 2 4 8 16 32 787200
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done



adb shell echo "979200  >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_big_979200"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./benchmark_batchsize_freq.sh ${NetName} 1 2 4 8 16 32 979200
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

  

adb shell echo "1036800  >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_big_1036800"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./benchmark_batchsize_freq.sh ${NetName} 1 2 4 8 16 32 1036800
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

 

adb shell echo "1248000  >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_big_1248000"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./benchmark_batchsize_freq.sh ${NetName} 1 2 4 8 16 32 1248000
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done



adb shell echo "1401600  >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_big_1401600"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./benchmark_batchsize_freq.sh ${NetName} 1 2 4 8 16 32 1401600
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

  

adb shell echo "1555200  >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_big_1555200"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./benchmark_batchsize_freq.sh ${NetName} 1 2 4 8 16 32 1555200
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

 

adb shell echo "1766400  >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_big_1766400"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./benchmark_batchsize_freq.sh ${NetName} 1 2 4 8 16 32 1766400
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

 

adb shell echo "1900800  >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_big_1900800"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./benchmark_batchsize_freq.sh ${NetName} 1 2 4 8 16 32 1900800
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

  

adb shell echo "2073600  >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_big_2073600"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./benchmark_batchsize_freq.sh ${NetName} 1 2 4 8 16 32 2073600
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

adb shell echo "2131200  >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_big_2131200"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./benchmark_batchsize_freq.sh ${NetName} 1 2 4 8 16 32 2131200
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done



adb shell echo "2208000  >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_big_2208000"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./benchmark_batchsize_freq.sh ${NetName} 1 2 4 8 16 32 2208000
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done






# small of haven't test

cat ./mnist.txt > ../source/demo/MnistBenchmarkUtils.cpp
cat ./mobile.txt > ../source/demo/mobilenetV2BenchmarkUtils.cpp

adb shell echo "1248000 >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_1248000"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./a1248000.sh ${NetName} 1 2 4 8 16 32 -p
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

sleep 200

adb shell echo "768000 >> /data/local/tmp/control.txt"

DeviceName="RedmiNote9P_768000"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./a768000.sh ${NetName} 1 2 4 8 16 32 -p
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

sleep 200

adb shell echo "576000 >> /data/local/tmp/control.txt"


DeviceName="RedmiNote9P_576000"

# echo "pid=\`pgrep usage\` ; su -c kill -9 \$pid" > ./cmd.txt # adb shell中连续执行多条指令的办法

for NetName in ${NetNames[@]}
do
    ./a576000.sh ${NetName} 1 2 4 8 16 32 -p
    # adb pull /data/local/tmp/train_bench.result ./${DeviceName}/train_bench/train_bench_${NetName}.result
    # adb shell < ./cmd.txt 
    adb pull /data/local/tmp/usage_monitor.result ./${DeviceName}/usage_monitor/usage_monitor_${NetName}.result
    sleep 30 # cooling device to avoid freq downgrade
    # python ./dataprocess.py ${DeviceName} ${NetName}
done

sleep 200