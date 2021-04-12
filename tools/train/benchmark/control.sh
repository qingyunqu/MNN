echo > control.txt
while true 
do
    control=`cat control.txt`
    
    if [ control -ne 0 ]; then
        echo change freq to $control
        freq=$control
		su -c echo 'userspace' > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
        su -c echo 'userspace' > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
        su -c echo 'userspace' > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
        su -c echo 'userspace' > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
        su -c echo 'userspace' > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
        su -c echo 'userspace' > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor
        su -c echo 'userspace' > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
        su -c echo 'userspace' > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
        su -c echo $freq > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
        su -c echo $freq > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
        su -c echo $freq > /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed
        su -c echo $freq > /sys/devices/system/cpu/cpu3/cpufreq/scaling_setspeed
        su -c echo $freq > /sys/devices/system/cpu/cpu4/cpufreq/scaling_setspeed
        su -c echo $freq > /sys/devices/system/cpu/cpu5/cpufreq/scaling_setspeed
        su -c echo $freq > /sys/devices/system/cpu/cpu6/cpufreq/scaling_setspeed
        su -c echo $freq > /sys/devices/system/cpu/cpu7/cpufreq/scaling_setspeed
        su -c kill -9 `pgrep -f usage`
        ./usage_monitor.sh usage_monitor.result 1 &
        echo > control.txt
	fi

done


#  echo 'schedutil' > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
#  echo 'schedutil' > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
#  echo 'schedutil' > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
#  echo 'schedutil' > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
#  echo 'schedutil' > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
#  echo 'schedutil' > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor
#  echo 'schedutil' > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
#  echo 'schedutil' > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor


#high
 echo 'userspace' > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
 echo 1785600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
 echo 1785600 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
 echo 1785600 > /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed
 echo 1785600 > /sys/devices/system/cpu/cpu3/cpufreq/scaling_setspeed
 echo 2419200 > /sys/devices/system/cpu/cpu4/cpufreq/scaling_setspeed
 echo 2419200 > /sys/devices/system/cpu/cpu5/cpufreq/scaling_setspeed
 echo 2419200 > /sys/devices/system/cpu/cpu6/cpufreq/scaling_setspeed
 echo 2419200 > /sys/devices/system/cpu/cpu7/cpufreq/scaling_setspeed


 #medium
 echo 'userspace' > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
 echo 1209600 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
 echo 1209600 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
 echo 1209600 > /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed
 echo 1209600 > /sys/devices/system/cpu/cpu3/cpufreq/scaling_setspeed
 echo 1612800 > /sys/devices/system/cpu/cpu4/cpufreq/scaling_setspeed
 echo 1612800 > /sys/devices/system/cpu/cpu5/cpufreq/scaling_setspeed
 echo 1612800 > /sys/devices/system/cpu/cpu6/cpufreq/scaling_setspeed
 echo 1612800 > /sys/devices/system/cpu/cpu7/cpufreq/scaling_setspeed

 #low
 echo 'userspace' > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
 echo 'userspace' > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
 echo 576000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
 echo 576000 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
 echo 576000 > /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed
 echo 576000 > /sys/devices/system/cpu/cpu3/cpufreq/scaling_setspeed
 echo 710400 > /sys/devices/system/cpu/cpu4/cpufreq/scaling_setspeed
 echo 710400 > /sys/devices/system/cpu/cpu5/cpufreq/scaling_setspeed
 echo 710400 > /sys/devices/system/cpu/cpu6/cpufreq/scaling_setspeed
 echo 710400 > /sys/devices/system/cpu/cpu7/cpufreq/scaling_setspeed