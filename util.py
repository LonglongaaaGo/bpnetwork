import numpy as np


def shuffle(feature,label):
    ''' get the shuffle feature and label
    :param feature: the input data (num, feature)
    :param label: the lable (num, 1)
    :return: shuffle_feature(num,feature),shuffle_label(num,1)
    '''
    m = feature.shape[0]
    permutation = list(np.random.permutation(m))
    shuffle_feature = feature[permutation,:]
    shuffle_label = label[permutation,:]
    return shuffle_feature,shuffle_label

def load_data(file_name):
    '''
    数据导入函数
    :param file_name: (string)训练数据位置
    :return: feature_data(mat)特征
            lable_data(mat)标签
    '''
    fr = open(file_name)
    feature_data =[];
    lable_data = [];
    for line in fr.readlines():
        curLine = []
        lineArr = line.split('\t')
        for i in range(0,2):
            curLine.append(float(lineArr[i]))
        feature_data.append(curLine)
        if len(lineArr)<3:
            continue;
        tempLine = []
        for i in range(2,3):
            tempLine.append(int(lineArr[i]))
        lable_data.append(tempLine)
    feature_mat = np.array(feature_data,dtype=float)
    label_mat = np.array(lable_data,dtype=int)
    fr.close()
    return feature_mat,label_mat

def trainsform_(label_mat,outvec):
    label_out_mat = np.zeros((label_mat.shape[0],outvec))

    for i in range(label_mat.shape[0]):
        label_out_mat[i][label_mat[i][0]] = 1
    return label_out_mat

if __name__ == "__main__":
    feature_mat, label_mat = load_data("train_data.txt")
    print(feature_mat)
    print(label_mat)

    print(feature_mat.shape)
    print(label_mat.shape)

    out_label = trainsform_(label_mat,4)
    print(out_label)
    print(out_label.shape)