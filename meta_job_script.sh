N_GPU=${1?At least 1}
for i in {0..3}
do
  sh job_script.sh $N_GPU $i
done
wait
