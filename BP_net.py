import numpy as np

class BP:
    def __init__(self,input,hin,layerNum,output):
        np.random.seed(1)
        self.layerNum = layerNum
        self.input = input
        self.output = output

        self.parameter = {}
        for i in range(1,layerNum+1):
            if i==1:
                W = np.random.rand(input, hin) * 0.01
                b = np.zeros((1, 1))
                actfun = "relu"
            elif i==layerNum:
                W = np.random.rand(hin, output) * 0.01
                b = np.zeros((1, 1))
                actfun = "softmax"
            else:
                W = np.random.rand(hin, hin) * 0.01
                b = np.zeros((1, 1))
                actfun = "relu"
            self.parameter["{}_W".format(i)] = W
            self.parameter["{}_b".format(i)] = b
            self.parameter["{}_fun".format(i)] = actfun

    def fun_(self,input,funName):

        if funName == "relu":
            out,cache = relu(input)

        elif funName == "softmax":
            out = softmax(input)
        return out

    def getScore(self,input,label):
        out = self.forward(input)
        sum = 0
        for i in range(len(label)):
            outlabel = np.where(out[i]==max(out[i]))
            if outlabel == label[i]:
                sum= sum+1;
        score = float(sum)/len(label)

        return score

    def forward(self,input):
        output = input
        self.parameter["{}_funout".format(0)] = input
        for i in range(1,self.layerNum+1):
            W = self.parameter["{}_W".format(i)]
            b = self.parameter["{}_b".format(i)]
            fun = self.parameter["{}_fun".format(i)]
            output = output.dot(W)
            self.parameter["{}_Wout".format(i)] = output
            output = output+b
            self.parameter["{}_bout".format(i)] = output
            output = self.fun_(output,fun)
            self.parameter["{}_funout".format(i)] = output
        return output

    def backward(self,output,label):

        d_A = output
        for i in range(self.layerNum ,0,-1):
            W = self.parameter["{}_W".format(i)]
            Z = self.parameter["{}_bout".format(i)]
            pre_A = self.parameter["{}_funout".format(i-1)]

            if self.parameter["{}_fun".format(i)] == "relu":
                d_Z = relu_backward(d_A, Z)
            else:
                d_Z = backward_softmax_with_loss(d_A, label)

            d_W =np.dot(pre_A.T,d_Z)
            d_b = sum(d_Z)
            d_pre_A = np.dot(d_Z,W.T)
            d_A = d_pre_A
            self.parameter["{}_dW".format(i)] = d_W
            self.parameter["{}_db".format(i)] = d_b

    def update(self,learning_rate):
        for i in range(1,self.layerNum+1):
            W = self.parameter["{}_W".format(i)]
            b = self.parameter["{}_b".format(i)]
            d_W = self.parameter["{}_dW".format(i)]
            d_b = self.parameter["{}_db".format(i)]
            self.parameter["{}_W".format(i)] = W -learning_rate*d_W
            self.parameter["{}_b".format(i)] = b -learning_rate*d_b

    def propagation(self,input,label,learning = 0.1,epoch = 100):

        for i in range(epoch):
            output = self.forward(input)
            self.backward(output,label)
            self.update(learning)
            loss = cross_entropy(output,label)
            print("loss:{}".format(loss))



def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)
    return dZ


def softmax(input):
    out = input.copy()
    out = np.exp(out)
    for i in range(input.shape[0]):
        temp_sum = np.sum(out[i])
        out[i] = out[i]/temp_sum
    return out

def cross_entropy(output,label):
    loss = 0
    for i in range(output.shape[0]):
        loss += -output[i][label[i]] + np.log(np.sum(np.exp(output[i])))
    loss=loss/float(output.shape[0])
    return loss

def backward_softmax_with_loss(output,label):
    d_J = output
    for i in range(output.shape[0]):
        # print(d_J[i][label[i]])
        d_J[i][label[i]] =d_J[i][label[i]] - 1
    d_J = np.divide(d_J,output.shape[0])
    return d_J
