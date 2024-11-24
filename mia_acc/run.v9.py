version="v9"

# suffix_epochs=50
savefolder="save_folder"
prefix_noise_type="mg"

for dataset in ["p100"]:
    for model in ["2f"]:
        for prefix_noise_params in list(range(1)):
            for suffix_epochs in [10, 20, 30, 40, 50, 100, 200, 500, 1000]:
                print(f"python run_mia_acc.{version}.py {dataset} {model} {suffix_epochs} {savefolder} {prefix_noise_type} {prefix_noise_params} > logs/{version}.{prefix_noise_type}.{prefix_noise_params}.{dataset}.{model}.T_{suffix_epochs}.log")
                print("wait")