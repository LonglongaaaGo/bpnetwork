import BP_net as BP
import util as ut


def main():
    feature_mat, label_mat = ut.load_data("train_data.txt")
    # print(feature_mat)
    # print(label_mat)
    feature_mat,label_mat =ut.shuffle(feature_mat,label_mat)
    print(feature_mat.shape)
    print(label_mat.shape)
    label_mat = label_mat.squeeze()

    test_num = int(0.2*feature_mat.shape[0])
    train_mat = feature_mat[test_num:]
    train_label = label_mat[test_num:]
    test_mat = feature_mat[0:test_num]
    test_label = label_mat[0:test_num]

    bp_net = BP.BP(input = feature_mat.shape[1],hin = 16,layerNum = 3,output = 4)

    bp_net.propagation(train_mat,train_label,learning = 0.1,epoch = 10000)

    score = bp_net.getScore(test_mat,test_label)
    print("score:{}".format(score))


if __name__ == "__main__":
    main()
