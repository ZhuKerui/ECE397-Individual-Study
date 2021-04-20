# Setup parameters
file_in="$1"
file_out="$2"
core_num=8
cmd='java -Xmx800m -jar ollie-app-latest.jar ${sub_file_in} -o ${sub_file_out} --output-format interactive &'

# Split the file
file_in_line=`awk 'END{print NR}' ${file_in}`
sub_file_line=`expr \( ${file_in_line} / ${core_num} \) + 1`
sub_file_name="${file_in%.*}_temp_"
split -l ${sub_file_line} ${file_in} -d -a 3 ${sub_file_name}

# Handle each sub-file
sub_out_name="${file_in%.*}_out"
sub_files=`ls ${sub_file_name}*`
file_count=0
start_time=`date`
:>t.log
for sub_file_in in ${sub_files}
do
    sub_file_out="${sub_out_name}_${file_count}"
    eval $cmd >> t.log 2>&1
    ((file_count++))
done
wait

# Merge the outputs and delete the temp files
cat ${sub_out_name}_* > ${file_out}
rm -rf ${sub_out_name}_*
rm -rf ${sub_file_name}*

# Send the email
end_time=`date`
echo "Task is done.\nStarting time: ${start_time};\nEnd time: ${end_time}" | mail -s "Task done" -A t.log keruiz2@illinois.edu
