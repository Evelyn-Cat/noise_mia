import sys
import pandas as pd
import matplotlib.pyplot as plt

sigma = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else None

filepath = f"log.p100.2f.C_1.gaussian.{sigma}.T_100.log"
filepath_png = f"p100.2f.C_1.gaussian.T_100.sigma_{sigma}.png"
print(f"filepath: {filepath}")
print(f"sigma: {sigma}")

with open(filepath, 'r', encoding='utf-8') as f:
    rs = f.readlines()

df = pd.DataFrame([], columns = ["train_loss", "train_accuracy", "val_loss", "val_accuracy"])

cnt = 0
lines, epochs = [], []
for r in rs:
    if "Epoch" in r:
        epochs.append(r.strip().split(" "))
    elif "/step - loss" in r:
        line = r.strip().split(" ")
        # train_loss, train_accuracy, val_loss, val_accuracy = line[7], "{:.2f}".format(float(line[10])), line[13], "{:.2f}".format(float(line[16]))
        df.loc[cnt, ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]] = line[7], "{:.2f}".format(float(line[10])*100), line[13], "{:.2f}".format(float(line[16])*100)
        cnt = cnt + 1

df = df.astype('float')

acc = sorted(df['val_accuracy'].tolist(), reverse=True)[0]
print(f"val accuracy: {acc}")

if mode == None:
    val_color, train_color = "green", "red"
    loss_linewidth, acc_linewidth = 1, 2

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['train_accuracy'], label='Train Accuracy', color=train_color, linewidth=acc_linewidth)
    ax1.plot(df['val_accuracy'], label='Validation Accuracy', color=val_color, linewidth=acc_linewidth)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.tick_params(axis='y')


    ax2 = ax1.twinx()
    ax2.plot(df['train_loss'], label='Train Loss', color=train_color, linestyle='--', linewidth=loss_linewidth)
    ax2.plot(df['val_loss'], label='Validation Loss', color=val_color, linestyle='--', linewidth=loss_linewidth)
    ax2.set_ylabel('Loss')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 10)


    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')


    plt.title('Training and Validation Metrics with Dual Y-Axes')
    ax1.grid(True, which='both', axis='both', linestyle='--')

    plt.savefig(filepath_png)
    print(f"file saved: {filepath_png}")
