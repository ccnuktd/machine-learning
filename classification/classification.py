import numpy as np
eps = 1e-8
np.random.seed(0)#same random number
X_train_fpath = "./data/X_train"
Y_train_fpath = "./data/Y_train"
X_test_fpath = "./data/X_test"
output_fpath = "./data/output_{}.csv"#print output
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f],dtype=float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f],dtype=float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f],dtype=float)
#函数用于标准化
def _normalize(X,train=True,specified_column = None,X_mean = None,X_std = None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
        # 注意这个的specified_column是一个一维向量
        # arange 是步长
    if train:
        X_mean = np.mean(X[:,specified_column],0).reshape(1,-1)
        X_std = np.std(X[:,specified_column],0).reshape(1,-1)
    X[:,specified_column] = (X[:,specified_column]-X_mean)/(X_std + eps)
    return X,X_mean,X_std
#函数用于分割数据
def _train_dev_split(X,Y,dev_ratio = 0.25):
    train_set = int(len(X) * (1 - dev_ratio))
    return X[:train_set],X[train_set:],Y[:train_set],Y[train_set:]

#数据处理
Y_train = Y_train.reshape(Y_train.shape[0],1)
X_train,X_mean,X_std = _normalize(X_train)
X_test,_,_ = _normalize(X_test,train=False,X_mean = X_mean,X_std = X_std)

dev_ratio = 0.1
X_train,X_dev,Y_train,Y_dev = _train_dev_split(X_train,Y_train,dev_ratio = dev_ratio)
train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

def _shuffle(X,Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    #函数非常有意思，如果是给的是一维数组就打乱次序，如果是二维数组就打乱二维的次序，但是
    #保持一维的次序不边
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return X[randomize],Y[randomize]
def _sigmoid(z):
    #后面两个参数是最大最小值
    return np.clip(1/(1.0+np.exp(-z)),eps,1-eps)
def _f(X,w,b):
    return _sigmoid(np.dot(X,w) + b)
def _predict(X,w,b):
    return np.round(_f(X,w,b)).astype(np.int)
def _accuracy(Y_pred,Y_label):
    acc = 1-np.mean(np.abs(Y_pred-Y_label))
    return acc
def _cross_entropy_loss(Y_pred,Y_label):
    cross_entropy = -np.dot(np.log(Y_pred),Y_label) - np.dot(np.log(1-Y_pred),1-Y_label)
    return cross_entropy
def _gradient(X,Y_label,w,b):
    Y_pred = _f(X,w,b)
    pred_error = Y_label - Y_pred
    #print(pred_error.shape)
    w_grad = -np.dot(pred_error,X)
    #print("w_grad:" + str(w_grad.shape))
    b_grad = -np.sum(pred_error)
    return w_grad,b_grad


# Zero initialization for weights ans bias


# Some parameters for training
max_iter = 100
batch_size = 50
learning_rate = 0.18

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# Calcuate the number of parameter updates
step = 0

w = np.zeros((data_dim,1))
b = np.zeros((1,1))

# Iterative training
for epoch in range(max_iter):
    # Random shuffle at the begging of each epoch
    #X_train, Y_train = _shuffle(X_train, Y_train)
    adagrad_w = np.zeros((data_dim,1))
    #adagrad_b = np.zeros((batch_size,1))
    X = np.zeros((batch_size,data_dim))
    Y = np.zeros((batch_size,1))
    # Mini-batch training
    w_grad = np.zeros((data_dim,1))
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx * batch_size:(idx + 1) * batch_size]
        Y = Y_train[idx * batch_size:(idx + 1) * batch_size]
        # Compute the gradient
        Y_pred = _f(X,w,b)
        # print("Y:" + str((Y).shape))
        w_grad = -np.dot(X.T,Y - _f(X,w,b))#b = 0
        # print("w_grad:" + str(w_grad.shape))
        adagrad_w += w_grad ** 2
        #adagrad_b += b_grad ** 2
        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate * w_grad / np.sqrt(adagrad_w + 0.0000000001)
        #b = b - learning_rate * b_grad.T / np.sqrt(adagrad_b.T) / step
    # Compute loss and accuracy of training set and development set
y_train_pred = _f(X_train, w, b)
Y_train_pred = np.round(y_train_pred)
train_acc.append(_accuracy(Y_train_pred, Y_train))
# train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

y_dev_pred = _f(X_dev, w, b)
Y_dev_pred = np.round(y_dev_pred)
dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
# dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

# print('Training loss: {}'.format(train_loss[-1]))
# print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))
import matplotlib.pyplot as plt
# Loss curve
# plt.plot(train_loss)
# plt.plot(dev_loss)
# plt.title('Loss')
# plt.legend(['train', 'dev'])
# plt.savefig('loss.png')
# plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()