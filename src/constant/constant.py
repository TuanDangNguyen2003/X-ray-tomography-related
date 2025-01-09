PATH_TSV_FOLDER_BASELINE_1 = "data/tsv/baseline_1"
PATH_TSV_FOLDER_BASELINE_2 = "data/tsv/baseline_2"
PATH_TSV_FOLDER_BASELINE_3 = "data/tsv/baseline_3"
PATH_TSV_FOLDER_BASELINE_4 = "data/tsv/baseline_4"
PATH_TSV_FOLDER_BASELINE_5 = "data/tsv/baseline_5"
PATH_TSV_FOLDER_BASELINE_6 = "data/tsv/baseline_6"
PATH_TSV_FOLDER_LOW_MAG_1 = "data/tsv/low_mag_1"
PATH_TSV_FOLDER_LOW_POW_1 = "data/tsv/low_pow_1"
PATH_TSV_FOLDER_LOW_MAG_2 = "data/tsv/low_mag_2"
PATH_TSV_FOLDER_LOW_POW_2 = "data/tsv/low_pow_2"

ALL_TSV_FOLDERS = [
    "data/tsv/baseline_1",
    "data/tsv/baseline_2",
    "data/tsv/baseline_3",
    "data/tsv/baseline_4",
    "data/tsv/baseline_5",
    "data/tsv/baseline_6",
    "data/tsv/low_mag_1",
    "data/tsv/low_pow_1",
    "data/tsv/low_mag_2",
    "data/tsv/low_pow_2",
]

COLOR_LIST_ALL_TSV_FOLDERS = [
    "red",  # baseline 1
    "blue",  # baseline 2
    "green",  # baseline 3
    "orange",  # baseline 4
    "purple",  # baseline 5
    "brown",  # baseline 6
    "black",  # low_mag_1
    "gray",  # low_pow_1
    "pink",  # low_mag_2
    "cyan",  # low_pow_2
]

MAX_NB_SCANS = 20
MIN_NB_SCANS = 15

X_VALUES_MAX = [i * 20 for i in range(MAX_NB_SCANS)]
X_VALUES_MIN = [i * 20 for i in range(MIN_NB_SCANS)]
