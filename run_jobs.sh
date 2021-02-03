#!/bin/bash
for i in {0..49}
do
   python serious_run.py --idx=$i --job_path="job"
   wait
done