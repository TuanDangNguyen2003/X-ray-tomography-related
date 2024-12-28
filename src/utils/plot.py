import matplotlib.pyplot as plt
import numpy as np


def plot_many_y_lists(
    x_values: list,
    y_values: list,
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

    plt.figure(figsize=(10, 6))  # Set figure size

    # Plot each y-values set with its corresponding label
    for y, label in zip(y_values, y_labels):
        plt.plot(x_values, y, label=label, marker="o", linestyle="-")

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
