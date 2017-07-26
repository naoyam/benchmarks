#!/usr/bin/env bash

p=$1
for i in $(seq 5); do
    echo Trial $i
    for exe in $(find . -name $p);  do
        echo "$exe"
        $exe > out
        if grep Accuracy out  | grep "e-07"; then
            echo $exe >> results.$p.$i
            cat out >> results.$p.$i
        else
            echo fail: $(grep Accuracy out)
        fi
    done
    echo ""
done
