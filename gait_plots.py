import matplotlib.pyplot as plt
import numpy as np

angles = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]

# ---------------- NM Condition ----------------
nm_data = {
    "GEINet":     [40.2, 38.9, 42.9, 45.6, 51.2, 42.0, 53.5, 57.6, 57.8, 51.8, 47.7],
    "CNN-LB":     [82.6, 90.3, 96.1, 94.3, 90.1, 87.4, 89.9, 94.0, 94.7, 91.3, 78.5],
    "GaitNet":    [93.1, 92.6, 90.8, 92.4, 87.6, 95.1, 94.2, 95.8, 92.6, 90.4, 90.2],
    "GaitSet":    [91.1, 99.0, 99.9, 97.8, 95.1, 94.5, 96.1, 98.3, 99.2, 98.1, 88.0],
    "PGOFI":      [91.2, 95.8, 96.6, 96.1, 96.0, 94.8, 94.9, 95.7, 94.6, 94.2, 92.8],
    "GaitPart":   [94.1, 98.6, 99.3, 98.5, 94.0, 92.3, 95.9, 98.4, 99.2, 97.8, 90.4],
    "STTN":       [95.6, 99.8, 100.0, 99.0, 97.3, 95.8, 97.6, 99.4, 99.7, 99.0, 93.5],
    "RDBA-Net(Ours)":   [98.9, 99.5, 99.6, 98.7, 96.3, 95.7, 97.4, 98.3, 99.2, 98.5, 99.3]
}

# ---------------- BG Condition ----------------
bg_data = {
    "GEINet":     [34.2, 29.3, 31.2, 35.2, 35.2, 27.6, 35.9, 43.5, 45.0, 39.0, 36.8],
    "CNN-LB":     [64.2, 80.6, 82.7, 76.9, 64.8, 63.1, 68.0, 76.9, 82.2, 75.4, 61.3],
    "GaitNet":    [88.8, 88.7, 88.7, 94.3, 85.4, 92.7, 91.1, 92.6, 84.9, 84.4, 86.7],
    "GaitSet":    [86.7, 94.2, 95.7, 93.4, 88.9, 85.5, 89.0, 91.7, 94.5, 95.9, 83.3],
    "PGOFI":      [87.6, 90.8, 91.7, 91.5, 91.0, 93.9, 90.1, 91.5, 92.0, 90.4, 89.5],
    "GaitPart":   [89.1, 94.8, 96.7, 95.1, 88.3, 84.9, 89.0, 93.5, 96.1, 93.8, 85.8],
    "STTN":       [92.4, 95.7, 97.0, 96.0, 92.5, 89.6, 91.7, 96.7, 98.8, 98.0, 88.5],
    "RDBA-Net(Ours)":   [92.0, 98.3, 98.2, 95.5, 93.2, 92.0, 93.1, 96.7, 98.1, 95.7, 89.0]
}

# ---------------- CL Condition ----------------
cl_data = {
    "GEINet":     [19.0, 20.3, 22.5, 23.5, 26.7, 21.3, 27.4, 28.2, 24.2, 22.5, 21.6],
    "CNN-LB":     [37.7, 57.2, 66.6, 61.1, 55.2, 54.6, 55.2, 59.1, 58.9, 48.8, 39.4],
    "GaitNet":    [50.1, 60.7, 72.4, 72.1, 74.6, 78.4, 70.3, 68.2, 53.5, 44.1, 40.8],
    "GaitSet":    [59.5, 75.0, 78.3, 74.6, 71.4, 71.3, 70.8, 74.1, 74.6, 69.4, 54.1],
    "PGOFI":      [73.0, 74.5, 79.1, 79.8, 81.5, 82.5, 81.1, 79.4, 77.8, 76.6, 75.7],
    "GaitPart":   [70.7, 85.5, 86.9, 83.3, 77.1, 72.5, 76.9, 82.2, 83.8, 80.2, 66.5],
    "STTN":       [69.7, 89.0, 88.4, 84.9, 78.8, 75.5, 79.2, 82.4, 82.6, 76.9, 61.9],
    "RDBA-Net(Ours)":   [75.1, 88.1, 89.8, 86.2, 81.1, 78.7, 80.1, 84.2, 86.2, 80.5, 69.3]
}


def plot_all_methods(condition_data, title):
    plt.figure(figsize=(10, 10))
    for method, values in condition_data.items():
        if method == "RDBA-Net(Ours)":
            plt.plot(angles, values, linestyle='-', linewidth=1.5, marker='o', label=method, color='blue')
        else:
            plt.plot(angles, values, linestyle='--', marker='o', label=method)

    plt.title(title, fontsize=16)
    plt.xlabel("Probe View Angle (°)", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=16)
    plt.yticks(np.arange(15, 105, 5), fontsize=15)
    plt.xticks(angles, fontsize=15)
    plt.grid(True, linestyle='--', linewidth=1)

    if "CL" in title:
        legend_loc = "upper center"
        legend_box = (0.5, 1)
    else:
        legend_loc = "lower center"
        legend_box = (0.5, 0)
    plt.legend(loc=legend_loc, bbox_to_anchor=legend_box,
               ncol=4, fontsize=13, frameon=True)

    plt.tight_layout(rect=[0, 0.01, 1, 1])
    plt.show()


# Plot all three
plot_all_methods(nm_data, "Accuracy vs. View Angle (NM Condition)")
plot_all_methods(bg_data, "Accuracy vs. View Angle (BG Condition)")
plot_all_methods(cl_data, "Accuracy vs. View Angle (CL Condition)")