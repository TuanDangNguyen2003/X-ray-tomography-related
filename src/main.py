import numpy as np

import constant.constant as const
import utils.fitFunctions as ff
import utils.plot as plot
import utils.utilFunctions as uf


def main():
    # Plot evolution of volumetric strain of all scanning sessions
    plot.plot_vol_strain_of_all_baseline(const.ALL_TSV_FOLDERS, const.MAX_NB_SCANS)
    return


if __name__ == "__main__":
    main()
