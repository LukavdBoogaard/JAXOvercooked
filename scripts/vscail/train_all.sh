#!/bin/bash
base_dir="${HOME}/overcook/JAXOvercooked"
log_dir="/data/${USER}/overcook_logs"


commands=(
  "bash $base_dir/scripts/vscail/run_expt.sh 7"
#  "bash $base_dir/scripts/vsc/run_ppo 6 SlipperyAnt-v5 10 2_000_000"
#  "bash $base_dir/scripts/vsc/run_ppo 5 SlipperyAnt-v5 8 2_000_000 10_000"
#  "bash $base_dir/scripts/vsc/run_ppo 4 SlipperyAnt-v5 10 2_000_000 10_000"
)
# order of args: script(0) cuda_device(1)
# old order: script(0) cuda_device(1) env(2) seed(3) friction_change_freq(4)


exp_outf="exp_mm_$(date +%Y-%m-%d_%H%M%S)"
exp_count=1
for command in "${commands[@]}"; do
  output_file="${log_dir}/console_outputs/${exp_outf}_exp${exp_count}.out"
  full_command="${command} ${output_file}"
  nohup $full_command > $output_file 2>&1 &

  echo "Running command: $command , output file: $output_file , experiment count: $exp_count"
  exp_count=$((exp_count+1))
done
