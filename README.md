### 广东省火灾预测研究

简介:
本项目用于广东省2026年国庆火灾发生情况预测。本文采用2025年对应日期的时间交叉验证方法，使用可变植被、固定人口和逐日气象信息，预测时使用历史植被人口信息，气象信息则由盘古气象大模型来产生。

### 项目构成
```utils```:该文件夹存储了数据的下载和处理方式，并将原始数据转为.npy文件，其中，ERA5的气象数据为全球数据，而其余数据为广东省数据

```Pangu-Weather-main```: 该文件夹下存储了盘古大模型的权重以及进行未来预测的脚本

```data```: 该文件夹存储了广东省的shp数据，下载的原始数据和经过处理之后的数据

```model```：存放了项目的非校准模型组和校准模型方法

项目运行流程：
1. 首先下载历史数据，在对于```utils```文件夹下的以```download```数据开头的脚本的时间脚本后，运行四个脚本(需要进行GEE的验证并指定GEE项目名称)
服务器中可直接获取2026年1月31日以前的数据
```
python ./utils/download_era5_pangu_format.py --start 2025-12-01 --end 2025-12-31 --out ./data/era5_grib
```
```
python ./utils/download_FIRMS.py
```
```
python ./utils/download_Lai_Fpar.py
```
```
python ./utils/download_Population.py
```
2. 进行气象数据的预测
首先下载运行的最近的全球ERA5气象数据
```
python ./utils/download_era5_pangu_format.py --start 2026-07-26 --end 2026-07-26 --out ./Pangu-Weather-main/era5_grib
```
然后将grib数据转化为npy数据以便处理:
```
python ./Pangu-Weather-main/era5_grib_to_pangu_npy.py
```
最后进行迭代预测
```
python ./Pangu-Weather-main/pangu_forecast_onestep_rolling.py
```
3. 对数据进行处理
以下的脚本将所有数据转化为1km数据并进行匹配
```
python process_data.py
```
4. 对模型进行训练
```
python main.py
```
5. 进行指标评估和预测
该文件夹下的内容需要在根目录下运行，而不是进入到特地文件夹后在运行。```workspace```中为不同的任务分配了不同的工作区，并存放下它们的对应结果。其中，如果出现某些脚本无法运行的情况，请务必检查是否前向已经运行了火灾预测的两个脚本。其中，每个脚本都配对了对应的校准和非校准版本，在使用时可以选择不同的checkpoint，历史预测和未来预测
