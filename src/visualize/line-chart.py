import matplotlib.pyplot as plt

def visualize(x_arrays, y_arrays, labels=None, title="", xlabel="", ylabel="", save_path=None, figsize=(8,5)):
    """
    Vẽ nhiều đường trên cùng 1 biểu đồ.

    Args:
        x_arrays (list of list/array): Mỗi phần tử là mảng trục X của 1 đường.
        y_arrays (list of list/array): Mỗi phần tử là mảng trục Y tương ứng.
        labels (list of str, optional): Tên nhãn cho từng đường.
        title (str, optional): Tiêu đề biểu đồ.
        xlabel (str, optional): Nhãn trục X.
        ylabel (str, optional): Nhãn trục Y.
        save_path (str, optional): Nếu có, lưu biểu đồ ra file.
        figsize (tuple, optional): Kích thước figure.
    """
    plt.figure(figsize=figsize)

    for i, (x, y) in enumerate(zip(x_arrays, y_arrays)):
        label = labels[i] if labels else None
        plt.plot(x, y, marker='o', label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels:
        plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()
