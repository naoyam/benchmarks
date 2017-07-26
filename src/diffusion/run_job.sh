#!/usr/bin/env bash

BATCH=batch.sh

rm -f $BATCH
WD=$(pwd)
TS=$(date +%Y%m%d_%H%M%S%N)
echo "#!/usr/bin/env bash" >> $BATCH
echo "#BSUB -G guests" >> $BATCH
echo "#BSUB -cwd $WD" >> $BATCH
echo "#BSUB -n 1" >> $BATCH
echo "#BSUB -q pdebug" >> $BATCH
echo "#BSUB -W 01:00" >> $BATCH
echo "#BSUB -o job_output.$TS.txt" >> $BATCH
echo "#BSUB -x" >> $BATCH

echo $*
for i in $*; do
	echo ./run2.sh $i >> $BATCH
done

echo "Submitting job"
bsub < $BATCH




