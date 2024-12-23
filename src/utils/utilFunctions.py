import numpy as np
import tifffile as tiff
import spam.deformation as defo
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import automationReg as reg
from functools import partial

def count_white_pixels(folder_path: str, img_volume_imgJ: list[int] , max_nb_scans: int):
    """
    Counts the number of white pixels in a 3D .tiff image stack.

    Args:
        folder_path (str): Path to the folder containing the .tiff images.
        img_volume_imgJ (list): List to store the volume of each image in ImageJ.
    
    Returns:
        int: Total number of white pixels across all slices.
    """
    for i in range(max_nb_scans):
        tiff_file = os.path.join(folder_path, f'Reslice of {i:02d}-reg-def.tif')
        # Load the TIFF stack
        image_stack = tiff.imread(tiff_file)
        # Count white pixels (value 255)
        white_pixel_count = np.sum(image_stack == 255)
        img_volume_imgJ.append(white_pixel_count)
    
    return 

def process_tsv(file_path: str):
    """
    Process a single TSV file and decompose Phi.

    returns: vol, z, t, r
    """

    # Load the TSV file
    tsv_data = pd.read_csv(file_path, sep='\t')

    # Clean up the column names by stripping extra spaces
    tsv_data.columns = tsv_data.columns.str.strip()

    # Extract the first (and only) row of data
    first_row = tsv_data.iloc[0]

    # Construct the Phi matrix
    Phi = np.array([
        [first_row['Fzz'], first_row['Fzy'], first_row['Fzx'], first_row['Zdisp']],
        [first_row['Fyz'], first_row['Fyy'], first_row['Fyx'], first_row['Ydisp']],
        [first_row['Fxz'], first_row['Fxy'], first_row['Fxx'], first_row['Xdisp']],
        [0, 0, 0, 1]
    ])

    # Decompose Phi using spam.deformation
    transformation = defo.decomposePhi(Phi)
    return transformation['vol'], transformation['z'], transformation['t'], transformation['r']  # Return both 'vol' and 'z' (zoom)

def plot_vol_strain_and_imgj_volume(img_data: list[int] , baseline_number_path: list[str], nb_scans: list[int]):
    if len(img_data) < 2:
        raise ValueError("At least two images are required for plotting.")
    
    # Calculate the difference in percentage from img00 for each image
    absolute_diff = [((img - img_data[0])/img_data[0]) * 100 for img in img_data]

    # Prepare lists for image labels, vol, and z values
    vol_values = [0]            

    # Set the interval in minutes between scans
    scan_interval = 20  # 20 minutes per scan

    # Prepare lists to hold the data for each scan
    all_vol_values = []
    scan_labels = []

    # Generate time labels based on the number of scans and interval
    time_labels = [i * scan_interval for i in range(nb_scans)]  # Starts from 0, increments by 20

    for scan_index, folder_path in enumerate(baseline_number_path):
        vol_values = [0]  # Initial vol value for the first scan point

        for i in range(1, nb_scans):
            file_name = f'00-{i:02d}-registration.tsv'
            file_path = os.path.join(folder_path, file_name)

            if os.path.exists(file_path):
                print(f"Processing file: {file_name} in Scan {scan_index + 1}")
                vol, _, _, _ = process_tsv(file_path)
                vol_values.append(vol * 100)  # Convert vol to percentage
            else:
                vol_values.append(np.nan)  # For padding if necessary

        # Store the results for each scan
        all_vol_values.append(vol_values)
        scan_labels.append('Evolution of Volumetric Strain')

    all_vol_values.append(absolute_diff)
    scan_labels.append('Difference in Volume')

    # Plot 'vol' evolution with time in minutes as the x-axis and add arrows
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for i, vol_values in enumerate(all_vol_values):
        ax1.plot(time_labels, vol_values, marker='o', linestyle='-', label=scan_labels[i])

    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Vol (%)', fontsize=12)
    plt.title('Evolution of Vol over Time (in %)', fontsize=14)

    # Adding arrows for x and y axes at the origin
    ax1.annotate('', xy=(time_labels[-1], 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax1.annotate('', xy=(0, max(max(vol_values) for vol_values in all_vol_values)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax1.set_xticks(time_labels)
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # Places the legend to the right
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjusts the plot area to fit the legend

    # Plot layout adjustments
    plt.tight_layout()
    plt.show()

'''
def square_root_fit(x_values, y_values):
    """
    Fits the data to the function y = a * sqrt(x) + b using linear regression.
    
    Parameters:
        x_values (list or numpy array): List of x-values (independent variable).
        y_values (list or numpy array): List of y-values (dependent variable).
    
    Returns:
        dict: A dictionary containing the optimized parameters, R-squared value, and fitted y-values.
    """
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    
    # Define the model function
    def model(x, a, b):
        return a * np.sqrt(x) + b
    
    # Fit the model to the data
    params, _ = curve_fit(model, x, y)
    a, b = params
    
    # Compute fitted values
    y_fit = model(x, a, b)
    
    # Calculate R-squared
    def r_squared(y_actual, y_fitted):
        residual = y_actual - y_fitted
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((y_actual - np.mean(y_actual))**2)
        return 1 - (ss_res / ss_tot)
    
    r2 = r_squared(y, y_fit)

    # Plot the data and the fit
    plt.subplots(figsize=(10, 6))
    plt.scatter(x, y, label="Volumetric strain of baseline 4", color="blue")
    plt.plot(x, y_fit, label=f"Fit: y = {a:.10f}√x + {b:.10f}", color="red")
    
    # Customizing the x-axis to display mapped x-values
    plt.xticks(x, labels=[f"{xi:.1f}" for xi in x], rotation=45)  # Rotates labels for better readability
    
    # Labels and title
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Volume (%)", fontsize=12)
    plt.title("Square Root Fit", fontsize=12)
    # Change size of the legend 
    plt.legend(fontsize=10)
    plt.grid(True)
    #plt.tight_layout()  # Ensure everything fits in the figure
    
    # Display the plot
    plt.show()
    
    # Return results
    return {
        "parameters": {
            "a": a,
            "b": b
        },
        "r_squared": r2,
        "fitted_y_values": y_fit.tolist()
    }
'''

'''def square_root_fit(x_values, y_values_list, baseline_nb_list: list[int]):
    """
    Fits multiple sets of y_values simultaneously to the function y = a * sqrt(x) + b using linear regression.
    
    Parameters:
        x_values (list or numpy array): List of x-values (independent variable).
        y_values_list (list of lists or 2D numpy array): List of y-values sets (dependent variable).
        baseline_nb_list (list): List of baseline numbers for labeling.
    
    Returns:
        dict: A dictionary containing the optimized parameters, R-squared value, and fitted y-values.
    """
    x = np.array(x_values, dtype=float)
    y_values_array = np.array(y_values_list, dtype=float)

    # Repeat x-values for all y-values to align dimensions
    x_combined = np.tile(x, y_values_array.shape[0])
    y_combined = y_values_array.flatten()

    # Define the model function
    def model(x, a, b):
        return a * np.sqrt(x) + b

    # Fit the model to the combined data
    params, _ = curve_fit(model, x_combined, y_combined)
    a, b = params

    # Compute fitted values for the combined data
    y_fit_combined = model(x_combined, a, b)

    # Calculate R-squared for the combined fit
    def r_squared(y_actual, y_fitted):
        residual = y_actual - y_fitted
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((y_actual - np.mean(y_actual))**2)
        return 1 - (ss_res / ss_tot)

    r2 = r_squared(y_combined, y_fit_combined)

    # Distinct color palette
    distinct_colors = [
        "red", "blue", "green", "orange", "purple", "brown", 
        "pink", "gray", "cyan", "magenta", "yellow"
    ]
    
    # Extend the color list if there are more data sets than colors
    while len(distinct_colors) < y_values_array.shape[0]:
        distinct_colors.extend(distinct_colors)

    # Plot the data and the fit
    plt.figure(figsize=(10, 6))
    
    # Scatter each set with a unique color
    for idx, y_values in enumerate(y_values_array):
        plt.scatter(x, y_values, label=f"Baseline {baseline_nb_list[idx]}", color=distinct_colors[idx], alpha=0.7)

    # Plot the fitted curve (single curve using unique x-values)
    x_fine = np.linspace(min(x), max(x), 500)  # More points for smoother curve
    y_fine_fit = model(x_fine, a, b)
    plt.plot(x_fine, y_fine_fit, label=f"Fit: y = {a:.10f}√x + {b:.10f}", color="black", linewidth=2)
    
    # Customizing the x-axis
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Volume (%)", fontsize=12)
    plt.title("Square Root Fit (Combined Data)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    # Display the plot
    plt.show()

    # Return results
    return {
        "parameters": {
            "a": a,
            "b": b
        },
        "r_squared": r2,
        "fitted_y_values": y_fit_combined.tolist()
    }'''

def square_root_fit(x_values, y_values_list, baseline_nb_list, plot=True):
    """
    Fits multiple sets of y_values simultaneously to the function y = a * sqrt(x) + b using linear regression.
    
    Parameters:
        x_values (list or numpy array): List of x-values (independent variable).
        y_values_list (list of lists or 2D numpy array): List of y-values sets (dependent variable).
        baseline_nb_list (list): List of baseline numbers for labeling.
    
    Returns:
        dict: A dictionary containing:
            - 'a': Fitted parameter for sqrt(x).
            - 'b': Fitted intercept parameter.
            - 'r_squared': R-squared value for the combined fit.
            - 'fitted_function': A callable function y(x) = a * sqrt(x) + b.
            - 'fitted_y_values': Fitted y-values for the combined data.
    """
    x = np.array(x_values, dtype=float)
    y_values_array = np.array(y_values_list, dtype=float)

    # Ensure input dimensions match
    if len(x) != y_values_array.shape[1]:
        raise ValueError("x_values length must match the number of columns in y_values_list.")

    # Combine data for fitting
    x_combined = np.tile(x, y_values_array.shape[0])
    y_combined = y_values_array.flatten()

    # Define the square root model function
    def model(x, a, b):
        return a * np.sqrt(x) + b

    # Fit the combined data
    params, _ = curve_fit(model, x_combined, y_combined)
    a, b = params

    # Compute fitted y-values
    y_fit_combined = model(x_combined, a, b)

    # Calculate R-squared
    def r_squared(y_actual, y_fitted):
        residual = y_actual - y_fitted
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((y_actual - np.mean(y_actual))**2)
        return 1 - (ss_res / ss_tot)

    r2 = r_squared(y_combined, y_fit_combined)

    # Create a parameterized function
    fitted_function = partial(model, a=a, b=b)
    
    if plot:
        # Plot the data and fit
        plt.figure(figsize=(10, 6))
        distinct_colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

        # Scatter plot for each dataset
        for idx, y_values in enumerate(y_values_array):
            plt.scatter(x, y_values, label=f"Baseline {baseline_nb_list[idx]}", color=distinct_colors[idx % 10], alpha=0.7)

        # Plot the fitted curve
        x_fine = np.linspace(min(x), max(x), 500)
        y_fine_fit = model(x_fine, a, b)
        plt.plot(x_fine, y_fine_fit, label=f"Fit: y = {a:.10f}√x + {b:.10f}", color="black", linewidth=2)

        # Customize the plot
        plt.xlabel("Time (minutes)", fontsize=12)
        plt.ylabel("Volume (%)", fontsize=12)
        plt.title("Square Root Fit (Combined Data)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        # Display the plot
        plt.show()

    # Return results
    return {
        "parameters": {
            "a": a,
            "b": b
        },
        "function_string": f"y = {a:.10f}√x + {b:.10f}",
        "r_squared": r2,
        "fitted_function": fitted_function,
        "fitted_y_values": y_fit_combined.tolist()
    }

def plot_vol_strain_of_all_baseline(scan_folders: list[str] , max_nb_scans: int):
    # Prepare lists for image labels, vol, and z values
    vol_values = [0]             # Initial vol value at 0 for image 00

    # Set the interval in minutes between scans
    scan_interval = 20  # 20 minutes per scan

    # Prepare lists to hold the data for each scan
    all_vol_values = []
    scan_labels = []

    # Generate time labels based on the number of scans and interval
    time_labels = [i * scan_interval for i in range(max_nb_scans)]  # Starts from 0, increments by 20

    for scan_index, folder_path in enumerate(scan_folders):
        vol_values = [0]  # Initial vol value for the first scan point

        for i in range(1, max_nb_scans):
            file_name = f'00-{i:02d}-registration.tsv'
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
            scan_labels.append('Low Mag 1')
        elif scan_index == 7:
            scan_labels.append('Low Power 1')
        else:
            scan_labels.append(f'Baseline {scan_index + 1}')

    # Plot 'vol' evolution with time in minutes as the x-axis and add arrows
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for i, vol_values in enumerate(all_vol_values):
        ax1.plot(time_labels, vol_values, marker='o', linestyle='-', label=scan_labels[i])

    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Vol (%)', fontsize=12)
    plt.title('Evolution of Vol over Time (in %)', fontsize=14)

    # Adding arrows for x and y axes at the origin
    ax1.annotate('', xy=(time_labels[-1], 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax1.annotate('', xy=(0, max(max(vol_values) for vol_values in all_vol_values)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax1.set_xticks(time_labels)
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjusts the plot area to fit the legend

    # Plot layout adjustments
    plt.tight_layout()
    plt.show()

def plot_error_on(scan_folders: list[str], max_nb_scans: int ):
    """
    scan_folders: list of paths to baseline folder (in string)
    max_nb_scans: number of scans
    """

    # Function to process a single TSV file and decompose Phi
    def process_tsv_error(file_path):
        # Load the TSV file
        tsv_data = pd.read_csv(file_path, sep='\t')

        # Clean up the column names by stripping extra spaces
        tsv_data.columns = tsv_data.columns.str.strip()

        # Extract the first (and only) row of data
        first_row = tsv_data.iloc[0]

        return first_row['error'] # Return the error value 

    # Prepare lists for error values
    err_values = [0]             # Initial error value at 0 for image 00

    # Set the interval in minutes between scans
    scan_interval = 20  # 20 minutes per scan

    # Prepare lists to hold the data for each scan
    all_err_values = []
    scan_labels = []

    # Generate time labels based on the number of scans and interval
    time_labels = [i * scan_interval for i in range(max_nb_scans)]  # Starts from 0, increments by 20

    for scan_index, folder_path in enumerate(scan_folders):
        err_values = [0]  # Initial vol value for the first scan point

        for i in range(1, max_nb_scans):
            file_name = f'00-{i:02d}-registration.tsv'
            file_path = os.path.join(folder_path, file_name)

            if os.path.exists(file_path):
                print(f"Processing file: {file_name} in Scan {scan_index + 1}")
                err = process_tsv_error(file_path)
                err_values.append(err)  
            else:
                err_values.append(np.nan)  # For padding if necessary

        # Store the results for each scan
        all_err_values.append(err_values)
        scan_labels.append(f'Baseline {scan_index + 1}')

    # Plot 'vol' evolution with time in minutes as the x-axis and add arrows
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for i, err_values in enumerate(all_err_values):
        ax1.plot(time_labels, err_values, marker='o', linestyle='-', label=scan_labels[i])

    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    plt.title('Evolution of Error over Time', fontsize=14)

    # Adding arrows for x and y axes at the origin
    ax1.annotate('', xy=(time_labels[-1], 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax1.annotate('', xy=(0, max(max(vol_values) for vol_values in all_err_values)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax1.set_xticks(time_labels)
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # Places the legend to the right
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjusts the plot area to fit the legend

    # Plot layout adjustments
    plt.tight_layout()
    plt.show()

def run_reg_script_on(nb_scans: int, first_img_path: str, output_path: str):
    reg.run_reg_script_x_images(nb_scans, first_img_path, output_path)

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
        raise ValueError("x_values length must match the number of columns in y_values_list.")

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
        ss_tot = np.sum((y_values - np.mean(y_values))**2)
        r2 = 1 - (ss_res / ss_tot)

        # Create a parameterized function
        fitted_function = partial(model, a=a, b=b)

        results.append({
            "a": a,
            "b": b,
            "r_squared": r2,
            "fitted_function": fitted_function,
            "fitted_y_values": y_fitted.tolist(),
            "function_string": f"y = {a:.10f}log(x) + {b:.10f}"
        })

    if plot:
        # Plot the data and fits
        plt.figure(figsize=(10, 6))
        distinct_colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

        for idx, (y_values, result) in enumerate(zip(y_values_array, results)):
            # Scatter plot for actual data
            plt.scatter(x, y_values, label=f"Baseline {baseline_nb_list[idx]}", color=distinct_colors[idx % 10], alpha=0.7)

            # Plot the fitted curve
            x_fine = np.linspace(min(x), max(x), 500)
            y_fine_fit = result["fitted_function"](x_fine)
            plt.plot(x_fine, y_fine_fit, label=f"Fit: y = {a:.10f}log(x) + {b:.10f}", color=distinct_colors[idx % 10], linewidth=2)

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
        raise ValueError("x_values length must match the number of columns in y_values_list.")

    if np.any(x <= 0):
        raise ValueError("x_values must be greater than 0 for the exponential function.")

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
        ss_tot = np.sum((y_values - np.mean(y_values))**2)
        r2 = 1 - (ss_res / ss_tot)

        # Create a parameterized function
        fitted_function = partial(model, a=a, b=b)

        results.append({
            "a": a,
            "b": b,
            "r_squared": r2,
            "fitted_function": fitted_function,
            "fitted_y_values": y_fitted.tolist(),
            "function_string": f"y = {a:.10f} * exp(-{b:.10f} * x)"
        })

    if plot:
        # Plot the data and fits
        plt.figure(figsize=(10, 6))
        distinct_colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

        for idx, (y_values, result) in enumerate(zip(y_values_array, results)):
            # Scatter plot for actual data
            plt.scatter(x, y_values, label=f"Baseline {baseline_nb_list[idx]}", color=distinct_colors[idx % 10], alpha=0.7)

            # Plot the fitted curve
            x_fine = np.linspace(min(x), max(x), 500)
            y_fine_fit = result["fitted_function"](x_fine)
            plt.plot(x_fine, y_fine_fit, label=f"Fit: y = {result['function_string']}", color=distinct_colors[idx % 10], linewidth=2)

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

def plot_3_fits_function(x_values: list[int], y_values, function_1, function_2, function_3, baseline_nb: int, sqrt_func: str, log_func: str, exp_func: str):
    """
    Plots two functions and the original y_values on the same plot.
    
    Parameters:
        x_values (list or numpy array): List of x-values.
        y_values (list or numpy array): Original y-values to plot as connected points.
        function_1 (callable): First function to plot.
        function_2 (callable): Second function to plot.
        function_3 (callable): Third function to plot.
    """
    # Ensure x_values and y_values are numpy arrays
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    
    # Evaluate the first function on all x_values
    y1 = function_1(x)
    
    # Filter out x = 0 for the second function
    valid_x = x[x > 0]  # Keep x-values greater than 0
    y2 = function_2(valid_x)

    y3 = function_3(x)
    
    # Plot the functions and original values
    plt.subplots(figsize=(10, 6))
    
    plt.plot(x, y, label=f"Baseline {baseline_nb}", color="red", linewidth=2, marker='o')
    
    # Square root fit function
    plt.plot(x, y1, label=sqrt_func, color="blue", linewidth=2)
    
    # Logarithm fit function
    plt.plot(valid_x, y2, label=log_func, color="green", linewidth=2)

    # Exponential fit function
    plt.plot(x, y3, label=exp_func, color="purple", linewidth=2)
    
    # Customize the plot
    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Vol (%)", fontsize=12)
    plt.title("Comparison of Original Data, Square Root Fit, and Logarithm Fit", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def horizontal_mvm_src_object(x1, x2, zoom_lv_z):
    return (x1 - (x2/ ((1 + x2/x1) * zoom_lv_z - 1)))

def plot_many_y_lists(x_values: list, y_values: list, x_label: str, y_labels: list, general_yLabel: str, title: str):
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
        raise ValueError("The number of y-value sets must match the number of y_labels.")
    
    plt.figure(figsize=(10, 6))  # Set figure size
    
    # Plot each y-values set with its corresponding label
    for y, label in zip(y_values, y_labels):
        plt.plot(x_values, y, label=label, marker='o', linestyle='-')
    
    plt.xticks(x_values)  # Set x-ticks to match the x-values
    plt.xlabel(x_label)
    plt.ylabel(general_yLabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)  # Add grid for better readability
    plt.show()

import numpy as np
from scipy.optimize import curve_fit
from functools import partial
import matplotlib.pyplot as plt

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
        raise ValueError("x_values length must match the number of columns in y_values_list.")

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
        ss_tot = np.sum((y_values - np.mean(y_values))**2)
        r2 = 1 - (ss_res / ss_tot)

        # Create a parameterized function
        fitted_function = partial(model, a=a, b=b)

        results.append({
            "a": a,
            "b": b,
            "r_squared": r2,
            "fitted_function": fitted_function,
            "fitted_y_values": y_fitted.tolist(),
            "function_string": f"y = {a:.10f} * (1 - e^(-{b:.10f} * x))"
        })

    if plot:
        # Plot the data and fits
        plt.figure(figsize=(10, 6))
        distinct_colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

        for idx, (y_values, result) in enumerate(zip(y_values_array, results)):
            # Scatter plot for actual data
            plt.scatter(x, y_values, label=f"Baseline {baseline_nb_list[idx]}", color=distinct_colors[idx % 10], alpha=0.7)

            # Plot the fitted curve
            x_fine = np.linspace(min(x), max(x), 500)
            y_fine_fit = result["fitted_function"](x_fine)
            plt.plot(x_fine, y_fine_fit, label=f"Fit: y = {result['function_string']}", color=distinct_colors[idx % 10], linewidth=2)

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

    