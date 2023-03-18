"""
draw the train_loss and val_loss of MyNet model
"""

import matplotlib.pyplot as plt


def plot_loss(result):
    all_tra_loss, all_val_loss = [], []
    wt_tra_loss, wt_val_loss = [], []
    et_tra_loss, et_val_loss = [], []
    tc_tra_loss, tc_val_loss = [], []

    fontsize = 13
    cuxi = 2
    train_color = '#8C1F66'
    val_color = '#0F6466'

    for j in range(0, len(result), 2):
        # print(result1[i].rstrip('\n').split(' ')[2])
        all_tra_loss.append(float(result[j].rstrip('\n').split(' ')[2]))
        all_val_loss.append(float(result[j].rstrip('\n').split(' ')[4]))

    for j in range(1, len(result), 2):
        # print(result1[i].rstrip('\n').split(' ')[2])
        et_tra_loss.append(float(result[j].rstrip('\n').split(' ')[1]))
        et_val_loss.append(float(result[j].rstrip('\n').split(' ')[8]))

    for j in range(1, len(result), 2):
        # print(result1[i].rstrip('\n').split(' ')[2])
        wt_tra_loss.append(float(result[j].rstrip('\n').split(' ')[5]))
        wt_val_loss.append(float(result[j].rstrip('\n').split(' ')[12]))

    for j in range(1, len(result), 2):
        # print(result1[i].rstrip('\n').split(' ')[2])
        tc_tra_loss.append(float(result[j].rstrip('\n').split(' ')[3]))
        tc_val_loss.append(float(result[j].rstrip('\n').split(' ')[10]))

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    ax[0][0].plot(range(0, len(all_tra_loss)), all_tra_loss, color=train_color, label='train_loss', linewidth=cuxi)
    ax[0][0].plot(range(0, len(all_val_loss)), all_val_loss, color=val_color, label='val_loss', linewidth=cuxi)
    ax[0][0].set_xticks(range(0, len(all_tra_loss)+1, 10))
    # ax[0][0].set_xlabel('Epoch', fontsize=fontsize, weight='bold')
    ax[0][0].set_ylabel('Loss', fontsize=fontsize, weight='bold')
    ax[0][0].legend(loc="upper right", fontsize=fontsize)

    ax[0][1].plot(range(0, len(et_tra_loss)), et_tra_loss, color=train_color, label='train_loss', linewidth=cuxi)
    ax[0][1].plot(range(0, len(et_val_loss)), et_val_loss, color=val_color, label='val_loss', linewidth=cuxi)
    ax[0][1].set_xticks(range(0, len(all_tra_loss)+1, 10))
    # ax[0][1].set_xlabel('Epoch', fontsize=fontsize, weight='bold')
    ax[0][1].set_ylabel('Dice(ET)', fontsize=fontsize, weight='bold')
    ax[0][1].legend(loc="lower right", fontsize=fontsize)

    ax[1][0].plot(range(0, len(wt_tra_loss)), wt_tra_loss, color=train_color, label='train_loss', linewidth=cuxi)
    ax[1][0].plot(range(0, len(wt_val_loss)), wt_val_loss, color=val_color, label='val_loss', linewidth=cuxi)
    ax[1][0].set_xticks(range(0, len(wt_tra_loss)+1, 10))
    # ax[1][0].set_xlabel('Epoch', fontsize=fontsize, weight='bold')
    ax[1][0].set_ylabel('Dice(WT)', fontsize=fontsize, weight='bold')
    ax[1][0].legend(loc="lower right", fontsize=fontsize)

    ax[1][1].plot(range(0, len(tc_tra_loss)), tc_tra_loss, color=train_color, label='train_loss', linewidth=cuxi)
    ax[1][1].plot(range(0, len(tc_val_loss)), tc_val_loss, color=val_color, label='val_loss', linewidth=cuxi)
    ax[1][1].set_xticks(range(0, len(tc_tra_loss)+1, 10))
    # ax[1][1].set_xlabel('Epoch', fontsize=fontsize, weight='bold')
    ax[1][1].set_ylabel('Dice(TC)', fontsize=fontsize, weight='bold')
    ax[1][1].legend(loc="lower right", fontsize=fontsize)

    # plt.grid()
    plt.show()


if __name__ == "__main__":
    data_path1 = './MyNet.txt'

    with open(data_path1, 'r') as f:
        data1 = f.readlines()

    print(data1)
    plot_loss(data1)



