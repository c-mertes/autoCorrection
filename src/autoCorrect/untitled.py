def computeLog2foldChange(data):
    log2folds = np.zeros_like(data, dtype=np.float)
    for i in range(0,data.shape[0]):
            for j in range(0,data.shape[0]):
                if data[i][j] == 0:
                    log2folds[i][j] = np.log2((data[i][j]+1)/np.mean(data[:,j]))
                else:
                    log2folds[i][j] = np.log2(data[i][j]/np.mean(data[:,j]))
    return log2folds