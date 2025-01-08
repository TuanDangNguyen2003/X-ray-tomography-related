import os

import matplotlib.pyplot as plt
import numpy as np

from utils.utilFunctions import process_tsv


def plot_many_y_lists(
    x_values: list[int],
    y_values: list[list],
    color_list: list[str],
    x_label: str,
    y_labels: list,
    general_yLabel: str,
    title: str,
):
    """
    Plots multiple lists of y-values against a single list of x-values on the same plot.

    Parameters:
        x_values (list or numpy array): List of x-values.
        y_values (list of lists or numpy arrays): List of lists of y-values.
        color_list (list of str): List of colors for each y-values set.
        y_labels (list): List of labels for each y-values set.
        x_label (str): Label for the x-axis.
        general_yLabel (str): General label for the y-axis.
        title (str): Title of the plot.
    """
    # Check if the number of y-value sets matches the number of labels
    if len(y_values) != len(y_labels):
        raise ValueError(
            "The number of y-value sets must match the number of y_labels."
        )

    # Check if the number of y-value sets matches the number of colors
    if len(y_values) != len(color_list):
        raise ValueError("The number of y-value sets must match the number of colors.")

    # Check if any of the y_values lists are equal in length to the x_values list, if not, add nan values
    for y in y_values:
        if len(y) != len(x_values):
            y.extend([np.nan] * (len(x_values) - len(y)))

    plt.figure(figsize=(10, 6))  # Set figure size

    # Plot each y-values set with its corresponding label and color
    for y, label, color in zip(y_values, y_labels, color_list):
        plt.plot(x_values, y, label=label, color=color, marker="o")

    """
    for y, label in zip(y_values, y_labels):
        plt.plot(x_values, y, label=label, marker="o", linestyle="-")
    """

    plt.xticks(x_values)  # Set x-ticks to match the x-values
    plt.xlabel(x_label)
    plt.ylabel(general_yLabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)  # Add grid for better readability
    plt.show()


def plot_functions(
    x_values: list[int],
    y_values,
    functions: list[callable],
    baseline_nb: int,
    function_labels: list[str],
    title: str,
):
    """
    Plots a list of functions and the original y_values on the same plot.

    Parameters:
        x_values (list or numpy array): List of x-values.
        y_values (list or numpy array): Original y-values to plot as connected points.
        functions (list of callable): List of functions to plot.
        baseline_nb (int): Identifier for the baseline dataset.
        function_labels (list of str): Labels for the functions being plotted.
        title (str): Title of the plot.
    """
    # Ensure x_values and y_values are numpy arrays
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)

    # Initialize the plot
    plt.subplots(figsize=(10, 6))

    # Plot the original y_values
    plt.plot(
        x, y, label=f"Baseline {baseline_nb}", color="red", linewidth=2, marker="o"
    )

    # Loop through the functions and plot them
    for func, label in zip(functions, function_labels):
        # Handle cases where function is undefined for certain x-values
        if label.lower().startswith("log"):  # Example: Skip x <= 0 for logarithm
            valid_x = x[x > 0]
            y_values = func(valid_x)
            plt.plot(valid_x, y_values, label=label, linewidth=2)
        else:
            y_values = func(x)
            plt.plot(x, y_values, label=label, linewidth=2)

    # Customize the plot
    plt.xticks(x)
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Vol (%)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_vol_strain_of_folders(
    scan_folders: list[str], max_nb_scans: int, color_list=list[str]
):
    """
    Plots the evolution of volumetric strain for all baseline scans.

    Parameters:
        scan_folders (list of str): List of paths to baseline folders.
        max_nb_scans (int): Maximum number of scans to plot.
        color_list (list of str): List of colors for each baseline scan.
    """
    # Prepare lists for image labels, vol, and z values
    vol_values = [0]  # Initial vol value at 0 for image 00

    # Set the interval in minutes between scans
    scan_interval = 20  # 20 minutes per scan

    # Prepare lists to hold the data for each scan
    all_vol_values = []
    scan_labels = []

    # Generate time labels based on the number of scans and interval
    time_labels = [
        i * scan_interval for i in range(max_nb_scans)
    ]  # Starts from 0, increments by 20

    for scan_index, folder_path in enumerate(scan_folders):
        vol_values = [0]  # Initial vol value for the first scan point

        for i in range(1, max_nb_scans):
            file_name = f"00-{i:02d}-registration.tsv"
            file_path = os.path.join(folder_path, file_name)

            if os.path.exists(file_path):
                print(f"Processing file: {file_name} in Scan {scan_index + 1}")
                vol, _, _, _ = process_tsv(file_path)
                vol_values.append(vol * 100)  # Convert vol to percentage
            else:
                vol_values.append(np.nan)  # For padding if necessary

        # Store the results for each scan
        all_vol_values.append(vol_values)
        if scan_index == 6:
            scan_labels.append("Low Mag 1")
        elif scan_index == 7:
            scan_labels.append("Low Power 1")
        else:
            scan_labels.append(f"Baseline {scan_index + 1}")

    # Plot 'vol' evolution with time in minutes as the x-axis and add arrows
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for i, vol_values in enumerate(all_vol_values):
        ax1.plot(
            time_labels,
            vol_values,
            marker="o",
            linestyle="-",
            label=scan_labels[i],
            color=color_list[i],
        )

    ax1.set_xlabel("Time (minutes)", fontsize=12)
    ax1.set_ylabel("Vol (%)", fontsize=12)
    plt.title("Evolution of Vol over Time (in %)", fontsize=14)

    # Adding arrows for x and y axes at the origin
    ax1.annotate(
        "",
        xy=(time_labels[-1], 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )
    ax1.annotate(
        "",
        xy=(0, max(max(vol_values) for vol_values in all_vol_values)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )

    ax1.set_xticks(time_labels)
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjusts the plot area to fit the legend

    # Plot layout adjustments
    plt.tight_layout()
    plt.show()
