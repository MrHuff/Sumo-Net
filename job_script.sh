N_GPU=${1?At least 1}
job_nr=${2?At least 1}
for i in $(seq 1 $N_GPU)
do
  python serious_run.py --dataset=$job_nr --seed=$i --eval_metric=0 --total_epochs=250 --grid_size=1000 --test_grid_size=10000 --hyperits=30 --patience=25 --validation_interval=2 --loss_type=0 --total_nr_gpu=$N_GPU> job_$job_nr_seed_$i.out &
  sleep 20
done
wait
