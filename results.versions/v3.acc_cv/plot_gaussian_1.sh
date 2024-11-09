## plot figure for gaussian
dataset="p100"
model="2f"
T=100
l2_norm_clip=1
noise_type="gaussian"
sigmas=(0.05)

for sigma in "${sigmas[@]}"; do
    python acc_cv_1104.py \
        -dataset ${dataset} \
        -model ${model} \
        -l2_norm_clip ${l2_norm_clip} \
        -noise_type ${noise_type} \
        -noise_params ${sigma} \
        -microbatches 64 \
        -epochs ${T} \
        -batch_size 8 > log.$dataset.$model.C_$l2_norm_clip.$noise_type.$sigma.T_$T.log
done
