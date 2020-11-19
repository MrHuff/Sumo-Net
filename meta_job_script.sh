N_GPU=${1?At least 1}
for i in {1..7}
do
  sh job_script.sh $N_GPU $i
done
wait
