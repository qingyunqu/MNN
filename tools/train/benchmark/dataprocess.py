# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
np.__version__


# %%
# set parameters
DeviceName = "samsungS8p"
NetName = "Lenet"
BatchSize = [1, 2, 4, 8, 16, 32, 64, 128, 256]

path_to_usage_monitor = "./" + DeviceName + "/usage_monitor/usage_monitor_" + NetName + ".result"
path_to_processed_data = "./" + DeviceName + "/processed_data/processed_data_" + NetName + ".csv"
path_to_expr_bench = "./" + DeviceName + "/train_bench/train_bench_" + NetName + ".result"
path_to_session_bench = "./" + DeviceName + "/bench_session.result"
path_to_time_table = "./" + DeviceName + "/processed_data/time_table_" + NetName + ".csv"
path_to_performance = "./" + DeviceName + "/processed_data/performance_" + NetName + ".csv"


# %%
# load raw data and extract their attributes into csv file
raw_data = pd.read_table(path_to_usage_monitor, header=None, sep=",", engine='python')

attributes= ["stamp", "processPid", "VIRT", "RES", "SHR", "cpuUsage",              "battery_current", "battery_voltage", "usb_current", "usb_voltage" ]
processed_data = pd.DataFrame(index=attributes)

data = pd.Series(["nan"]*10, attributes)

count = 0
for line in raw_data[0]:
    if "NEW DATA" in line:
        count = count + 1
        data["stamp"] = line.split()[2]
    if "battery_current" in line:
        data["battery_current"] = line.split()[1]
    if "battery_voltage" in line:
        data["battery_voltage"] = line.split()[1]
    if "status of pid" in line:
        data["processPid"] = line.split()[3]
    if "shell" in line:
        data["VIRT"] = line.split()[4]
        data["RES"] = line.split()[5]
        data["SHR"] = line.split()[6]
        data["cpuUsage"] = line.split()[8]
    if "END" in line:
        processed_data[count] = data
        data[:] = "nan"

matrix = processed_data.T

matrix.to_csv(path_to_processed_data)

    


# %%
# running time of expr (train + infer) and session (infer)
expr_data = pd.read_table(path_to_expr_bench, header=None, delimiter="\t")
session_data = pd.read_table(path_to_session_bench, header=None, delimiter="\t", engine='python')

attributes= ["batchsize", "expr_train", "expr_infer", "session_infer"]
time = pd.DataFrame(index=attributes)
data = pd.Series(["nan"]*4, attributes)

count = 0
for batchsize in BatchSize:
    count = count + 1
    data["batchsize"] = batchsize
    keywords_expr = "(batchsize is " + str(batchsize) + ")"
    keywords_session = NetName + "_" + str(batchsize) + ".mnn"
    print(keywords_session)
    for line in expr_data[0]:
        #print(type(line))
        if keywords_expr in line:
            if "Training" in line:
                data["expr_train"] = line.split()[6]
            if "Inferring" in line:
                data["expr_infer"] = line.split()[6]
    for line in session_data[0]:
        if keywords_session in line:
            data["session_infer"] = line.split()[12]
    time[count] = data
    data[:] = "nan"

matrix = time.T
matrix.to_csv(path_to_time_table)


# %%
# read data from processed_data and analyse it with time_stamp
processed_data = pd.read_table(path_to_processed_data, header=0, index_col=0, sep=",")
attributes= ["batchsize", "VIRT", "RES", "cpuUsage_train", "cpuUsage_infer",             "battery_current", "battery_voltage", "usb_current", "usb_voltage" ]
performance = pd.DataFrame(index=attributes)
perform_temp = pd.Series([0]*9, attributes) # 初始化性能值存储条

# 记录训练、推断过程的时间戳
trainStart = 0
trainEnd = 0
inferStart = 0
inferEnd = 0

num = 0
for batchsize in BatchSize:
    num = num + 1
    path_to_stamp = "./" + DeviceName + "/train_stamp/" + NetName + "/" + "train_stamp_" + str(batchsize) + ".result" 
    data = pd.read_table(path_to_stamp, header=None, delimiter="\t")
    for line in data[0]:
        if "Begin training" in line:
            trainStart = int(line.split()[3])
        if "End trainning" in line:
            trainEnd = int(line.split()[3])
        if "Begin inferring" in line:
            inferStart = int(line.split()[3])
        if "End inferring" in line:
            inferEnd = int(line.split()[3])

    count = 0 # 用来记录一个阶段里出现的记录条数
    for indexs in processed_data.index: # 按行对数据进行遍历
        if int(processed_data.loc[indexs]["stamp"]) > trainStart and int(processed_data.loc[indexs]["stamp"]) < trainEnd:
            if str(processed_data.loc[indexs]["VIRT"]) == "nan": 
                continue
            count = count + 1 
            perform_temp["cpuUsage_train"] = perform_temp["cpuUsage_train"] + processed_data.loc[indexs]["cpuUsage"]
            # 内存数据有M后缀，需处理一下
            perform_temp["VIRT"] = max (int(processed_data.loc[indexs]["VIRT"].split("M")[0]), perform_temp["VIRT"])
            perform_temp["RES"] = max (int(processed_data.loc[indexs]["RES"].split("M")[0]), perform_temp["RES"])#
            # perform_temp["SHR"] = max (int(processed_data.loc[indexs]["SHR"].split(".")[0]), perform_temp["SHR"])
            perform_temp["battery_current"] = max (int(processed_data.loc[indexs]["battery_current"]), perform_temp["battery_current"])
            perform_temp["battery_voltage"] = max (int(processed_data.loc[indexs]["battery_voltage"]), perform_temp["battery_voltage"])
            # perform_temp["usb_current"] = max (processed_data.loc[indexs]["usb_current"], perform_temp["usb_current"])
            # perform_temp["usb_current"] = max (processed_data.loc[indexs]["usb_current"], perform_temp["usb_current"])
    perform_temp["cpuUsage_train"] = perform_temp["cpuUsage_train"]/count

    count = 0 
    for indexs in processed_data.index: # 按行对数据进行遍历
        if int(processed_data.loc[indexs]["stamp"]) > inferStart and int(processed_data.loc[indexs]["stamp"]) < inferEnd:
            if str(processed_data.loc[indexs]["VIRT"]) == "nan": 
                continue
            count = count + 1 
            perform_temp["cpuUsage_infer"] = perform_temp["cpuUsage_infer"] + processed_data.loc[indexs]["cpuUsage"]
            perform_temp["VIRT"] = max (int(processed_data.loc[indexs]["VIRT"].split("M")[0]), perform_temp["VIRT"])
            perform_temp["RES"] = max (int(processed_data.loc[indexs]["RES"].split("M")[0]), perform_temp["RES"])
            # perform_temp["SHR"] = max (float(processed_data.loc[indexs]["SHR"].split("M")[0]), perform_temp["SHR"])
            perform_temp["battery_current"] = max (int(processed_data.loc[indexs]["battery_current"]), perform_temp["battery_current"])
            perform_temp["battery_voltage"] = max (int(processed_data.loc[indexs]["battery_voltage"]), perform_temp["battery_voltage"])
            # perform_temp["usb_current"] = max (processed_data.loc[indexs]["usb_current"], perform_temp["usb_current"])
            # perform_temp["usb_current"] = max (processed_data.loc[indexs]["usb_current"], perform_temp["usb_current"])
    perform_temp["cpuUsage_infer"] = perform_temp["cpuUsage_infer"]/count
    
    performance[num] = perform_temp
    perform_temp[:] = 0

    matrix = performance.T
    matrix.to_csv(path_to_performance)


# %%
matrix


# %%
data = pd.read_table(path_to_processed_data, header=0, index_col=0, sep=",")
data.loc[1]["stamp"]


# %%
x = "float"
x = float(x)
x


