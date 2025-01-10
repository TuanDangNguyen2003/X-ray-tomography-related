import numpy as np

import constant.constant as const
import utils.fitFunctions as ff
import utils.plot as plot
import utils.utilFunctions as uf


def main():
    """
    # Plot evolution of volumetric strain of all scanning sessions
    plot.plot_vol_strain_of_folders(
        const.ALL_TSV_FOLDERS, const.MAX_NB_SCANS, const.COLOR_LIST_ALL_TSV_FOLDERS
    )

    # Plot the evolution of volumetric strain for baseline 3, 4 scans
    vol_values = uf.get_vol_from_tsv_folders(
        [
            const.PATH_TSV_FOLDER_LOW_MAG_2,
            const.PATH_TSV_FOLDER_LOW_POW_1,
            const.PATH_TSV_FOLDER_LOW_MAG_1,
            const.PATH_TSV_FOLDER_LOW_POW_2,
        ],
        const.MAX_NB_SCANS,
    )

    plot.plot_many_y_lists(
        const.X_VALUES_MAX,
        vol_values,
        ["red", "blue", "green", "orange"],
        "Time (minutes)",
        ["Low Mag 2", "Low Pow 1", "Low Mag 1", "Low Pow 2"],
        "Volumetric Strain (%)",
        "Volumetric Strain Evolution",
    )
    """
    vol_values = uf.get_vol_from_tsv_folders(
        [const.PATH_TSV_FOLDER_BASELINE_3], const.MAX_NB_SCANS
    )

    # Remove element vol_values[0][0] because it is 0
    for i in range(len(vol_values)):
        vol_values[i] = np.delete(vol_values[i], 0)

    ff.logarithm_fit(const.X_VALUES_MAX_EXC_0, vol_values, [3])

    """ Fit using square root function and logarithm function 
    x_values = [20 * i for i in range(20)]
    x_values_1 = [20 * i for i in range(15)]
    y_baseline_1 = []
    y_baseline_3 = []
    y_baseline_4 = []
    y_baseline_5 = []
    y_baseline_6 = []

    y_baseline_1.append(0)
    y_baseline_3.append(0)
    y_baseline_4.append(0)
    y_baseline_5.append(0)
    y_baseline_6.append(0)

    for i in range(1, 20):
        file_name = f'00-{i:02d}-registration.tsv'
        regs_3_path = '/data/tuan/OPT_PRO/micro_baseline_3/' + file_name
        regs_4_path = '/data/tuan/OPT_PRO/micro_baseline_4/' + file_name
        regs_5_path = '/data/tuan/OPT_PRO/micro_baseline_5/' + file_name
        regs_6_path = '/data/tuan/OPT_PRO/micro_baseline_6/' + file_name
        vol_3, z_3, _ , _ = uf.process_tsv(regs_3_path)
        vol_4, z_4, _ , _ = uf.process_tsv(regs_4_path)
        vol_5, z_5, _ , _ = uf.process_tsv(regs_5_path)
        vol_6, z_6, _ , _ = uf.process_tsv(regs_6_path)
        y_baseline_3.append(vol_3 * 100)
        y_baseline_4.append(vol_4 * 100)
        y_baseline_5.append(vol_5 * 100)
        y_baseline_6.append(vol_6 * 100)

    str_func_3_sqt              = uf.exponential_fit(x_values,[y_baseline_3],[3], True)[0].get('function_string')
    str_func_4_sqt              = uf.exponential_fit(x_values,[y_baseline_4],[4], True)[0].get('function_string')
    str_func_5_sqt              = uf.exponential_fit(x_values,[y_baseline_5],[5], True)[0].get('function_string')
    str_func_6_sqt              = uf.exponential_fit(x_values,[y_baseline_6],[6], True)[0].get('function_string')
    """
    """
    x_values.remove(0)
    x_values_1.remove(0)
    y_baseline_log_1 = y_baseline_1.copy()
    y_baseline_log_1.remove(0)
    y_baseline_log_3 = y_baseline_3.copy()
    y_baseline_log_3.remove(0)
    y_baseline_log_4 = y_baseline_4.copy()
    y_baseline_log_4.remove(0)
    y_baseline_log_5 = y_baseline_5.copy()
    y_baseline_log_5.remove(0)
    y_baseline_log_6 = y_baseline_6.copy()
    y_baseline_log_6.remove(0)

    function_baseline_1_log = (uf.logarithm_fit(x_values_1,[y_baseline_log_1],[1], False))[0].get('fitted_function')
    str_func_1_log          = (uf.logarithm_fit(x_values_1,[y_baseline_log_1],[1], False))[0].get('function_string')
    function_baseline_3_log = (uf.logarithm_fit(x_values,[y_baseline_log_3],[3], False))[0].get('fitted_function') 
    str_func_3_log          = (uf.logarithm_fit(x_values,[y_baseline_log_3],[3], False))[0].get('function_string')
    function_baseline_4_log = (uf.logarithm_fit(x_values,[y_baseline_log_4],[4], False))[0].get('fitted_function')
    str_func_4_log          = (uf.logarithm_fit(x_values,[y_baseline_log_4],[4], False))[0].get('function_string')
    function_baseline_5_log = (uf.logarithm_fit(x_values,[y_baseline_log_5],[5], False))[0].get('fitted_function')
    str_func_5_log          = (uf.logarithm_fit(x_values,[y_baseline_log_5],[5], False))[0].get('function_string')
    function_baseline_6_log = (uf.logarithm_fit(x_values,[y_baseline_log_6],[6], False))[0].get('fitted_function')
    str_func_6_log          = (uf.logarithm_fit(x_values,[y_baseline_log_6],[6], False))[0].get('function_string')

    uf.plot_2_fits_function([20 * i for i in range(15)] , y_baseline_1, function_baseline_1_squared, function_baseline_1_log, 1, str_func_1_sqt, str_func_1_log)
    uf.plot_2_fits_function([20 * i for i in range(20)], y_baseline_3, function_baseline_3_squared, function_baseline_3_log, 3, str_func_3_sqt, str_func_3_log)
    uf.plot_2_fits_function([20 * i for i in range(20)], y_baseline_4, function_baseline_4_squared, function_baseline_4_log, 4, str_func_4_sqt, str_func_4_log)
    uf.plot_2_fits_function([20 * i for i in range(20)], y_baseline_5, function_baseline_5_squared, function_baseline_5_log, 5, str_func_5_sqt, str_func_5_log)
    uf.plot_2_fits_function([20 * i for i in range(20)], y_baseline_6, function_baseline_6_squared, function_baseline_6_log, 6, str_func_6_sqt, str_func_6_log)
    """

    return


if __name__ == "__main__":
    main()
