import os
import numpy as np
import cfgrib


# ============================================================
# 1. 只需要修改这里
# ============================================================
UPPER_GRIB_PATH = "./Pangu-Weather-main/era5_grib/input_upper_20261119.grib"
SURFACE_GRIB_PATH = "./Pangu-Weather-main/era5_grib/input_surface_20261119.grib"

OUTPUT_DIR = "./Pangu-Weather-main/era5_npy"

# 输出文件名
OUTPUT_UPPER_NAME = "input_upper_20261119.npy"
OUTPUT_SURFACE_NAME = "input_surface_20261119.npy"


# ============================================================
# 2. 从 GRIB 中读取指定变量
# ============================================================
def load_grib_variables(grib_path):
    """
    一个 GRIB 文件可能被 cfgrib 拆成多个 Dataset。
    这里将所有 Dataset 中的数据变量统一收集起来。
    """
    datasets = cfgrib.open_datasets(grib_path)

    variables = {}
    for ds in datasets:
        for name in ds.data_vars:
            variables[name] = ds[name]

    return variables


def to_numpy(data_array):
    """
    去掉单时次对应的长度为 1 的维度，并转换为 float32。
    """
    data_array = data_array.squeeze(drop=True)
    return np.asarray(data_array.values, dtype=np.float32)


# ============================================================
# 3. 主流程
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reading upper GRIB:", UPPER_GRIB_PATH)
    upper_vars = load_grib_variables(UPPER_GRIB_PATH)

    print("Upper variables:", list(upper_vars.keys()))

    # Pangu upper 变量顺序：
    # [z, q, t, u, v]
    upper = np.stack(
        [
            to_numpy(upper_vars["z"]),
            to_numpy(upper_vars["q"]),
            to_numpy(upper_vars["t"]),
            to_numpy(upper_vars["u"]),
            to_numpy(upper_vars["v"]),
        ],
        axis=0,
    )

    print("Reading surface GRIB:", SURFACE_GRIB_PATH)
    surface_vars = load_grib_variables(SURFACE_GRIB_PATH)

    print("Surface variables:", list(surface_vars.keys()))

    # ERA5 在 cfgrib 中通常将：
    # 10u -> u10
    # 10v -> v10
    # 2t  -> t2m
    #
    # Pangu surface 变量顺序：
    # [msl, 10u, 10v, 2t]
    surface = np.stack(
        [
            to_numpy(surface_vars["msl"]),
            to_numpy(surface_vars["u10"]),
            to_numpy(surface_vars["v10"]),
            to_numpy(surface_vars["t2m"]),
        ],
        axis=0,
    )

    upper_out = os.path.join(OUTPUT_DIR, OUTPUT_UPPER_NAME)
    surface_out = os.path.join(OUTPUT_DIR, OUTPUT_SURFACE_NAME)

    np.save(upper_out, upper.astype(np.float32))
    np.save(surface_out, surface.astype(np.float32))

    print()
    print("Conversion finished.")
    print("Upper shape:", upper.shape)
    print("Surface shape:", surface.shape)
    print("Saved:", upper_out)
    print("Saved:", surface_out)

    # 简单检查盘古全球输入形状
    expected_upper = (5, 13, 721, 1440)
    expected_surface = (4, 721, 1440)

    if upper.shape != expected_upper:
        print(
            "[WARNING] upper shape is not the standard Pangu shape:",
            expected_upper,
        )

    if surface.shape != expected_surface:
        print(
            "[WARNING] surface shape is not the standard Pangu shape:",
            expected_surface,
        )


if __name__ == "__main__":
    main()