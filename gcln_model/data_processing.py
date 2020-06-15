from sklearn.preprocessing import normalize
import numpy as np

def data_normalize(data):
    data = 10 * normalize(data, norm='l2', axis=1)
    return data

def constant_check(data, var_names):  # O(mn) time. m: number of points, n: number of terms.
    const_dict = {}
    assert data.shape[1] == len(var_names)
    for column in range(len(var_names)):
        if var_names[column] in ['1', '(* 1)']:  # the all-one column representing bias
            continue
        if (np.max(data[:, column]) == np.min(data[:, column])):  # This column contains a constant
            const_dict[column] = np.max(data[:, column])
    data = np.delete(data, list(const_dict.keys()), 1)
    named_const_dict = {var_names[key]:value for key,value in const_dict.items()}
    for i in sorted(const_dict.keys(), reverse=True):   # update variable list
        del var_names[i]
    return data, named_const_dict


def redundancy_check(data, var_names):   # O(mn^2) time. m: number of points, n: number of terms.
    assert data.shape[1] == len(var_names)
    named_redundancy_list, columns_to_remove = [], []
    for index1 in range(len(var_names)):
        for index2 in range(index1 + 1, len(var_names)):
            if np.max(np.abs(data[:, index1] - data[:, index2])) == 0:   # redundant term found
                columns_to_remove.append(index2)
                named_redundancy_list.append((var_names[index1], var_names[index2]))
    columns_to_remove = list(set(columns_to_remove))   # deduplicate
    data = np.delete(data, columns_to_remove, 1)
    for i in sorted(columns_to_remove, reverse=True):   # update variable list
        del(var_names[i])
    return data, named_redundancy_list


def threshold_pruning(data, var_names, pruning_threshold):   # O(mnt) time. t: number of terms to prun.
    assert data.shape[1] == len(var_names)
    data_abs = np.abs(data)
    named_pruned_list = []
    while len(var_names) >= 2:
        pruned = False
        for row in data_abs:
            max_index = np.argmax(row)
            second_max = np.partition(row, -2)[-2]
            if second_max < pruning_threshold * row[max_index]:
                named_pruned_list.append(var_names[max_index])
                del(var_names[max_index])
                data_abs = np.delete(data_abs, max_index, 1)
                data = np.delete(data, max_index, 1)
                pruned = True
                break
        if not pruned:
            break
    return data, named_pruned_list


def remove_big_value(data, data_cleaning_threshold):
    assert data_cleaning_threshold > 1
    row_to_remove = []
    for i in range(data.shape[0]):
        row_abs = np.abs(data[i])
        row_abs_nonzero = row_abs[row_abs!=0]
        if len(row_abs_nonzero) and np.max(row_abs_nonzero) > data_cleaning_threshold * np.min(row_abs_nonzero):
            row_to_remove.append(i)
    data = np.delete(data, row_to_remove, axis=0)
    return data
