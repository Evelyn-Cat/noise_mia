version="v4"

suffix_epochs=1000
savefolder="save_folder"
prefix_noise_type="mg"

for dataset in ["p100"]:
    for model in ["2f"]:
        for prefix_noise_params in list(range(90)):
            print(f"python run_mia_acc.py {dataset} {model} {suffix_epochs} {savefolder} {prefix_noise_type} {prefix_noise_params} > logs/{version}.{prefix_noise_type}.{prefix_noise_params}.{dataset}.{model}.T_{suffix_epochs}.log")
            print("wait")
