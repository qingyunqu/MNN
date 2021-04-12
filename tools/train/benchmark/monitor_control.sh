echo > control.txt
while true 
do
    control=`cat control.txt`
    
    if [ control -ne 0 ]; then
        echo change numCore to $control
        su -0 kill -9 `pgrep -f usage`
        ./usage_monitor.sh usage_monitor.result 1 &
        echo > control.txt
	fi

done