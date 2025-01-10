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

    return


if __name__ == "__main__":
    main()
