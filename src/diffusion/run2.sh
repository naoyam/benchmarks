#!/usr/bin/env bash

function detect_exe() {
	find . -name "diffusion_cuda*.exe" -exec basename {} \; | sort | uniq
}

if [ $# -eq 0 ]; then
	p=$(detect_exe)
elif [ $1 = "--find-exe" ]; then
	detect_exe
	exit 
elif [ $1 = "-p" ]; then
	p=$(detect_exe)
	p=$(echo $p | awk -v x=$2 '{print $x}')
else
	p=$*
fi

for e in $p; do
	echo "executable: $e"
	of=results.$p
	rm -f $of
	for i in $(seq 5); do
		echo Trial $i
		for exe in $(find . -name $e);  do
			echo "$exe"
			$exe > out
			if grep Accuracy out  | grep "e-07"; then
				echo $exe >> $of
				cat out >> $of
			else
				echo fail: $(grep Accuracy out)
			fi
		done
		echo ""
	done
done
