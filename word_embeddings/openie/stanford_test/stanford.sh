# Setup parameters
file_in=sub_sent0
file_out=sub_sent0_out.txt
cmd='python stanford.py ${file_in} ${file_out}'
start_time=`date`
:>t.log
eval $cmd >> t.log 2>&1
end_time=`date`
echo $start_time
echo $end_time
echo "task done" | mail -s "Task done" -A t.log keruiz2@illinois.edu
