list1=("fmnist") # "fmnist" epoch 25; p100 epoch 100
list2=("2f" "lr") # "lr"
# l2=(0.5 1 2 3 4 5) # 1
# noise_type=("gaussian")
# noise_params=(0.7243976515506757 0.6967240601856282) # 1 2 3 4 5 6 7 8 9
Ts=(1 5 10 20 50 100) # 1) # 
sens=(1 2 3 4 5)
noise_params=(1 2 3 4)
noise_type=("lmo")

for param in "${list1[@]}"; do
    for value in "${list2[@]}"; do
        for noise_param in "${noise_params[@]}"; do
            for T in "${Ts[@]}"; do
                for sen in "${sens[@]}"; do
                    for noise_type in "${noise_type[@]}"; do
                        # echo "Processing $param with $value"
                        python acc_cv.py -dataset ${param} -model ${value} -l2_norm_clip ${sen} -noise_type $noise_type -noise_params ${noise_param} -microbatches 25 -epochs $T -batch_size 25 > log.$noise_type.$param.$value.$noise_param.$T.sen=$sen.log
                    done
                done
            done
        done
    done
done
