version="v8"

# suffix_epochs=50
savefolder="save_folder"
prefix_noise_type="gaussian"
repeat=10

for dataset in ["p100"]:
    for model in ["2f"]:
        for suffix_epochs in [1000, 10, 20, 30, 40, 50, 100, 200, 500]:
            for prefix_noise_params in [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000]:
                print(f"python run_mia_acc.{version}.py {dataset} {model} {suffix_epochs} {savefolder} {prefix_noise_type} {prefix_noise_params} > logs/rp_{repeat}.{version}.{prefix_noise_type}.{prefix_noise_params}.{dataset}.{model}.T_{suffix_epochs}.log")
                print("wait")
