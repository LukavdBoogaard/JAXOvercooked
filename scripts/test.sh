seeds=(0 1 2) # 2 3 4)
task_ids=( use_task_id no-use_task_id)
multiheads=( use_multihead no-use_multihead)
# layer_norms=( no-use-layer-norm use-layer-norm)
shared_backbone=(shared_backbone no-shared_backbone)
cnn=(use_cnn no-use_cnn)
layouts=(
  "easy_levels"
)

tid=$SLURM_ARRAY_TASK_ID  
n_seeds=${#seeds[@]}
n_task_ids=${#task_ids[@]}
n_multiheads=${#multiheads[@]}
# n_layer_norms=${#layer_norms[@]}
n_shared_backbone=${#shared_backbone[@]}
n_layouts=${#layouts[@]}
n_cnn=${#cnn[@]}

index() {                            # index tid base₀ base₁ …
    local val=$1; shift
    local bases=("$@")
    local i
    for (( i=0; i<${#bases[@]}; i++ )); do
        local stride=1
        for (( j=i+1; j<${#bases[@]}; j++ )); do
            stride=$(( stride * bases[j] ))
        done
        printf '%s ' $(( val / stride ))
        val=$(( val % stride ))
    done
}
