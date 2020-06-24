import numpy as np

# x
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):       
        subset = seq[i : (i + size)]           
        aaa.append([item for item in subset])  
        #aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

#--------------------------------------------------
# xy
def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:]
        tmp_y = dataset[x_end_number : y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

#-----------------------------------------------
# xy n행마다 자르는 곳 리셋
def split_xy2(dataset, time_steps, y_column, n):
    x_data, y_data = list(), list()
    for j in np.arange(0, len(dataset), n):
        x, y = list(), list()
        for i in range(0, n):
            start = i+j
            x_end_number = start + time_steps
            y_end_number = x_end_number + y_column
            tmp_x = dataset[start : x_end_number, :]
            tmp_y = dataset[x_end_number : y_end_number, :]
            x.append(tmp_x)
            y.append(tmp_y)
        x_data.append(x)
        y_data.append(y)   
    return np.array(x_data), np.array(y_data)