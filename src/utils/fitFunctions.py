from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def exponential_fit(x_values, y_values_list, baseline_nb_list: list[int], plot=True):
    """
    Fits multiple sets of y_values simultaneously to the function y = a * (1 - e^(-b * x)) using nonlinear regression.

    Parameters:
        x_values (list or numpy array): List of x-values (independent variable).
        y_values_list (list of lists or 2D numpy array): List of y-values sets (dependent variable).
        baseline_nb_list (list of int): List of baseline numbers for labeling in the plot.
        plot (bool): Whether to plot the data and fits. Defaults to True.

    Returns:
        list: A list of dictionaries, each containing:
            - 'a': Coefficient for the model.
            - 'b': Exponential rate parameter.
            - 'r_squared': R-squared value.
            - 'fitted_function': A function y(x) = a * (1 - e^(-b * x)) for new x-values.
            - 'fitted_y_values': Fitted y-values for the dataset.
    """
    # Ensure x_values and y_values_list are numpy arrays
    x = np.array(x_values, dtype=float)
    y_values_array = np.array(y_values_list, dtype=float)

    if len(x) != y_values_array.shape[1]:
        raise ValueError(
            "x_values length must match the number of columns in y_values_list."
        )

    # Define the exponential model function
    def model(x, a, b):
        return a * (1 - np.exp(-b * x))

    # Fit the model to each dataset and calculate R-squared
    results = []
    for idx, y_values in enumerate(y_values_array):
        # Fit the curve using nonlinear regression
        params, _ = curve_fit(model, x, y_values, p0=[1.0, 0.1], maxfev=10000)
        a, b = params

        # Compute fitted values
        y_fitted = model(x, a, b)

        # Calculate R-squared
        residual = y_values - y_fitted
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Create a parameterized function
        fitted_function = partial(model, a=a, b=b)

        results.append(
            {
                "a": a,
                "b": b,
                "r_squared": r2,
                "fitted_function": fitted_function,
                "fitted_y_values": y_fitted.tolist(),
                "function_string": f"y = {a:.10f} * (1 - e^(-{b:.10f} * x))",
            }
        )

    if plot:
        # Plot the data and fits
        plt.figure(figsize=(10, 6))
        distinct_colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

        for idx, (y_values, result) in enumerate(zip(y_values_array, results)):
            # Scatter plot for actual data
            plt.scatter(
                x,
                y_values,
                label=f"Baseline {baseline_nb_list[idx]}",
                color=distinct_colors[idx % 10],
                alpha=0.7,
            )

            # Plot the fitted curve
            x_fine = np.linspace(min(x), max(x), 500)
            y_fine_fit = result["fitted_function"](x_fine)
            plt.plot(
                x_fine,
                y_fine_fit,
                label=f"Fit: y = {result['function_string']}",
                color=distinct_colors[idx % 10],
                linewidth=2,
            )

        # Customize the plot
        plt.xlabel("Time (minutes)", fontsize=12)
        plt.ylabel("Volume (%)", fontsize=12)
        plt.title("Exponential Fit", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        # Display the plot
        plt.show()

    return results


def logarithm_fit(x_values, y_values_list, baseline_nb_list: list[int], plot=True):
    """
    Fits multiple sets of y_values simultaneously to the function y = a * log(x) + b using linear regression.

    Parameters:
        x_values (list or numpy array): List of x-values (independent variable).
        y_values_list (list of lists or 2D numpy array): List of y-values sets (dependent variable).

    Returns:
        list: A list of dictionaries, each containing:
            - 'a': Coefficient for log(x).
            - 'b': Intercept.
            - 'r_squared': R-squared value.
            - 'fitted_function': A function y(x) = a * log(x) + b for new x-values.
            - 'fitted_y_values': Fitted y-values for the dataset.
    """
    # Ensure x_values and y_values_list are numpy arrays
    x = np.array(x_values, dtype=float)
    y_values_array = np.array(y_values_list, dtype=float)

    if len(x) != y_values_array.shape[1]:
        raise ValueError(
            "x_values length must match the number of columns in y_values_list."
        )

    if np.any(x <= 0):
        raise ValueError("x_values must be greater than 0 for the logarithm function.")

    # Define the logarithmic model function
    def model(x, a, b):
        return a * np.log(x) + b

    # Fit the model to each dataset and calculate R-squared
    results = []
    for idx, y_values in enumerate(y_values_array):
        params, _ = curve_fit(model, x, y_values)
        a, b = params

        # Compute fitted values
        y_fitted = model(x, a, b)

        # Calculate R-squared
        residual = y_values - y_fitted
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Create a parameterized function
        fitted_function = partial(model, a=a, b=b)

        results.append(
            {
                "a": a,
                "b": b,
                "r_squared": r2,
                "fitted_function": fitted_function,
                "fitted_y_values": y_fitted.tolist(),
                "function_string": f"y = {a:.10f}log(x) + {b:.10f}",
            }
        )

    if plot:
        # Plot the data and fits
        plt.figure(figsize=(10, 6))
        distinct_colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

        for idx, (y_values, result) in enumerate(zip(y_values_array, results)):
            # Scatter plot for actual data
            plt.scatter(
                x,
                y_values,
                label=f"Baseline {baseline_nb_list[idx]}",
                color=distinct_colors[idx % 10],
                alpha=0.7,
            )

            # Plot the fitted curve
            x_fine = np.linspace(min(x), max(x), 500)
            y_fine_fit = result["fitted_function"](x_fine)
            plt.plot(
                x_fine,
                y_fine_fit,
                label=f"Fit: y = {a:.10f}log(x) + {b:.10f}",
                color=distinct_colors[idx % 10],
                linewidth=2,
            )

        # Customize the plot
        plt.xlabel("Time (minutes)", fontsize=12)
        plt.ylabel("Volume (%)", fontsize=12)
        plt.title("Logarithmic Fit", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        # Display the plot
        plt.show()

    return results


def exponential_fit(x_values, y_values_list, baseline_nb_list: list[int], plot=True):
    """
    Fits multiple sets of y_values simultaneously to the function y = a * exp(-b*x) using nonlinear regression.

    Parameters:
        x_values (list or numpy array): List of x-values (independent variable).
        y_values_list (list of lists or 2D numpy array): List of y-values sets (dependent variable).
        baseline_nb_list (list[int]): Baseline identifiers for each y-value set.
        plot (bool): Whether to plot the results.

    Returns:
        list: A list of dictionaries, each containing:
            - 'a': Coefficient for the exponential function.
            - 'b': Exponent parameter.
            - 'r_squared': R-squared value.
            - 'fitted_function': A function y(x) = a * exp(-b*x) for new x-values.
            - 'fitted_y_values': Fitted y-values for the dataset.
    """
    # Ensure x_values and y_values_list are numpy arrays
    x = np.array(x_values, dtype=float)
    y_values_array = np.array(y_values_list, dtype=float)

    if len(x) != y_values_array.shape[1]:
        raise ValueError(
            "x_values length must match the number of columns in y_values_list."
        )

    if np.any(x <= 0):
        raise ValueError(
            "x_values must be greater than 0 for the exponential function."
        )

    # Define the exponential model function
    def model(x, a, b):
        return a * np.exp(-b * x)

    # Fit the model to each dataset and calculate R-squared
    results = []
    for idx, y_values in enumerate(y_values_array):
        params, _ = curve_fit(model, x, y_values, maxfev=10000)
        a, b = params

        # Compute fitted values
        y_fitted = model(x, a, b)

        # Calculate R-squared
        residual = y_values - y_fitted
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Create a parameterized function
        fitted_function = partial(model, a=a, b=b)

        results.append(
            {
                "a": a,
                "b": b,
                "r_squared": r2,
                "fitted_function": fitted_function,
                "fitted_y_values": y_fitted.tolist(),
                "function_string": f"y = {a:.10f} * exp(-{b:.10f} * x)",
            }
        )

    if plot:
        # Plot the data and fits
        plt.figure(figsize=(10, 6))
        distinct_colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

        for idx, (y_values, result) in enumerate(zip(y_values_array, results)):
            # Scatter plot for actual data
            plt.scatter(
                x,
                y_values,
                label=f"Baseline {baseline_nb_list[idx]}",
                color=distinct_colors[idx % 10],
                alpha=0.7,
            )

            # Plot the fitted curve
            x_fine = np.linspace(min(x), max(x), 500)
            y_fine_fit = result["fitted_function"](x_fine)
            plt.plot(
                x_fine,
                y_fine_fit,
                label=f"Fit: y = {result['function_string']}",
                color=distinct_colors[idx % 10],
                linewidth=2,
            )

        # Customize the plot
        plt.xlabel("Time (minutes)", fontsize=12)
        plt.ylabel("Volume (%)", fontsize=12)
        plt.title("Exponential Fit", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        # Display the plot
        plt.show()

    return results
