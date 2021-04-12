echo > control_big.txt
while true 
do
    control=`cat control_big.txt`
    
    if [ control -ne 0 ]; then
        echo change freq to $control
        freq=$control
		su -c echo 'userspace' > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
        su -c echo 'userspace' > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
        su -c echo $freq > /sys/devices/system/cpu/cpu6/cpufreq/scaling_setspeed
        su -c echo $freq > /sys/devices/system/cpu/cpu7/cpufreq/scaling_setspeed
        su -c kill -9 `pgrep -f usage`
        ./usage_monitor.sh usage_monitor.result 1 &
        echo > control_big.txt
	fi

done