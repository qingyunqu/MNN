adb shell "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies > /data/local/tmp/freq_list.result" 
adb pull /data/local/tmp/freq_list.result ./
 
freq_list=`cat freq_list.result`
echo $freq_list
 
adb shell "su -0 chmod 0777 /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor"
adb shell "echo 'userspace' > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor"
#  echo 'userspace' > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
#  echo 'userspace' > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
#  echo 'userspace' > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
#  echo 'userspace' > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
#  echo 'userspace' > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor
#  echo 'userspace' > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
#  echo 'userspace' > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
#  echo 1785600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#  echo 1785600 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#  echo 1785600 > /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed
#  echo 1785600 > /sys/devices/system/cpu/cpu3/cpufreq/scaling_setspeed
#  echo 2419200 > /sys/devices/system/cpu/cpu4/cpufreq/scaling_setspeed
#  echo 2419200 > /sys/devices/system/cpu/cpu5/cpufreq/scaling_setspeed
#  echo 2419200 > /sys/devices/system/cpu/cpu6/cpufreq/scaling_setspeed
#  echo 2419200 > /sys/devices/system/cpu/cpu7/cpufreq/scaling_setspeed