import matplotlib.pyplot as plt


def plot_train_loss(result):
    y = [[], [], [], []]
    label = ['Yi_Ding', 'MyNet', 'UNet', 'Zhengrong_Luo']
    for index, i in enumerate(result):
        print(index)
        for j in range(0, len(i), 2):
            # print(result1[i].rstrip('\n').split(' ')[2])
            y[index].append(float(result[index][j].rstrip('\n').split(' ')[2]))

    fig, ax = plt.subplots()
    for index, _ in enumerate(y):
        ax.plot(range(1, len(y[index])+1, 1), y[index], label=f'{label[index]}')

    ax.set_xticks(range(1, len(y[0])+1, 2))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE-DSC Loss')
    ax.set_title('Train Loss')
    ax.legend()

    plt.grid()
    plt.show()


if __name__ == "__main__":
    data_path1 = './Yi_Ding.txt'
    data_path2 = './MyNet.txt'
    data_path3 = './UNet.txt'
    data_path4 = './Zhengrong_Luo.txt'
    data_path5 = './LiuLiangLiang/txt'

    with open(data_path1, 'r') as f:
        data1 = f.readlines()

    with open(data_path2, 'r') as f:
        data2 = f.readlines()

    with open(data_path3, 'r') as f:
        data3 = f.readlines()

    with open(data_path4, 'r') as f:
        data4 = f.readlines()

    plot_train_loss([data1, data2, data3, data4])
    # plot_val_loss([data1, data2])

    #
    # with open(data_path2, 'r') as f:
    #     data = f.readlines()
    # plot_result(data)



