#!/bin/bash  

for num in {500..1000..2000}:
do
    echo $num
    ./vecadd00 $num
    sleep 10
    ./vecadd01 $num
done
