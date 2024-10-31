list1=("p100") # "fmnist" epoch 25; p100 epoch 100
list2=("2f" "lr")
# l2=(0.5 1 2 3 4 5) # 1
# noise_type=("gaussian")
noise_params=(0.5 1 2 3 4 5 6 7 8 9) # 1


for param in "${list1[@]}"; do
    for value in "${list2[@]}"; do
        for noise_param in "${noise_params[@]}"; do
            # echo "Processing $param with $value"
            python acc_cv.py -dataset ${param} -model ${value} -l2_norm_clip 1 -noise_type gaussian -noise_params ${noise_param} -microbatches 1 -epochs 100 -batch_size 250 > log.$param.$value.$noise_param.log
        done
    done
done