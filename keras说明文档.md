#1 快速开始：30s上手Keras

Keras的核心数据结构是“模型”，模型是一种组织网络层的方式。Keras中主要的模型是Sequential模型，Sequential是一系列网络层按顺序构成的栈。你也可以查看函数式模型来学习建立更复杂的模型

Sequential模型如下

```python
from keras.models import Sequential

model = Sequential()

```
将一些网络层通过`.add()`堆叠起来，就构成了一个模型：

```python
from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))
```
完成模型的搭建后，我们需要使用`.compile()`方法来编译模型：

```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```
编译模型时必须指明损失函数和优化器，如果你需要的话，也可以自己定制损失函数。Keras的一个核心理念就是简明易用，同时保证用户对Keras的绝对控制力度，用户可以根据自己的需要定制自己的模型、网络层，甚至修改源代码。

```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```
完成模型编译后，我们在训练数据上按batch进行一定次数的迭代来训练网络

```python
model.fit(x_train, y_train, epochs=5, batch_size=32)
```
当然，我们也可以手动将一个个batch的数据送入网络中训练，这时候需要使用：

```python
model.train_on_batch(x_batch, y_batch)
```
随后，我们可以使用一行代码对我们的模型进行评估，看看模型的指标是否满足我们的要求：

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```
或者，我们可以使用我们的模型，对新的数据进行预测：

```python
classes = model.predict(x_test, batch_size=128)
```
搭建一个问答系统、图像分类模型，或神经图灵机、word2vec词嵌入器就是这么快。支撑深度学习的基本想法本就是简单的，现在让我们把它的实现也变的简单起来！


#2 常见问题如下
##2.1 如何保存Keras模型？
我们不推荐使用pickle或cPickle来保存Keras模型

你可以使用`model.save(filepath)`将Keras模型和权重保存在一个HDF5文件中，该文件将包含：

* 模型的结构，以便重构该模型
* 模型的权重
* 训练配置（损失函数，优化器等）
* 优化器的状态，以便于从上次训练中断的地方开始

使用`keras.models.load_model(filepath)`来重新实例化你的模型，如果文件中存储了训练配置的话，该函数还会同时完成模型的编译

例子：

```pyhton
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```
如果你只是希望保存模型的结构，而不包含其权重或配置信息，可以使用：

```python
# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
```
这项操作将把模型序列化为json或yaml文件，这些文件对人而言也是友好的，如果需要的话你甚至可以手动打开这些文件并进行编辑。

当然，你也可以从保存好的json文件或yaml文件中载入模型：

```python
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML
model = model_from_yaml(yaml_string)
```
如果需要保存模型的权重，可通过下面的代码利用HDF5进行保存。注意，在使用前需要确保你已安装了HDF5和其Python库h5py

```python
model.save_weights('my_model_weights.h5')
```
如果你需要在代码中初始化一个完全相同的模型，请使用：

```python
model.load_weights('my_model_weights.h5')
```
如果你需要加载权重到不同的网络结构（有些层一样）中，例如fine-tune或transfer-learning，你可以通过层名字来加载模型：

```python
model.load_weights('my_model_weights.h5', by_name=True)
```
例如：

```python
"""
假如原模型为：
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="dense_1"))
    model.add(Dense(3, name="dense_2"))
    ...
    model.save_weights(fname)
"""
# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name="dense_1"))  # will be loaded
model.add(Dense(10, name="new_dense"))  # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)
```

##2.2 如何获取中间层的输出？
一种简单的方法是创建一个新的`Model`，使得它的输出是你想要的那个输出

```python
from keras.models import Model

model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(input=model.input,
                                 output=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
```
此外，我们也可以建立一个Keras的函数来达到这一目的：

```python
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([X])[0]
```
当然，我们也可以直接编写Theano和TensorFlow的函数来完成这件事

注意，如果你的模型在训练和测试两种模式下不完全一致，例如你的模型中含有Dropout层，批规范化（BatchNormalization）层等组件，你需要在函数中传递一个learning_phase的标记，像这样：

```python
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([X, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([X, 1])[0]
```

##2.3 当验证集的loss不再下降时，如何中断训练？
可以定义`EarlyStopping`来提前终止训练

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
```

##2.4 验证集是如何从训练集中分割出来的？
如果在`model.fit`中设置`validation_spilt`的值，则可将数据分为训练集和验证集，例如，设置该值为0.1，则训练集的最后10%数据将作为验证集，设置其他数字同理。注意，原数据在进行验证集分割前并没有被shuffle，所以这里的验证集严格的就是你输入数据最末的x%。


##2.5 训练数据在训练时会被随机洗乱吗？
是的，如果`model.fit`的`shuffle`参数为真，训练的数据就会被随机洗乱。不设置时默认为真。训练数据会在每个epoch的训练中都重新洗乱一次。

验证集的数据不会被洗乱

##2.6 如何在每个epoch后记录训练/测试的loss和正确率？

model.fit在运行结束后返回一个History对象，其中含有的history属性包含了训练过程中损失函数的值以及其他度量指标。

```python
hist = model.fit(X, y, validation_split=0.2)
print(hist.history)
```

##2.7 如何使用状态RNN（stateful RNN）？
一个RNN是状态RNN，意味着训练时每个batch的状态都会被重用于初始化下一个batch的初始状态。

当使用状态RNN时，有如下假设

* 所有的batch都具有相同数目的样本
* 如果`X1`和`X2`是两个相邻的batch，那么对于任何`i`，`X2[i]`都是`X1[i]`的后续序列
要使用状态RNN，我们需要

* 显式的指定每个batch的大小。可以通过模型的首层参数`batch_input_shape`来完成。`batch_input_shape`是一个整数tuple，例如(32,10,16)代表一个具有10个时间步，每步向量长为16，每32个样本构成一个batch的输入数据格式。
* 在RNN层中，设置`stateful=True`
要重置网络的状态，使用：

* `model.reset_states()`来重置网络中所有层的状态
* `layer.reset_states()`来重置指定层的状态

例子：

```python
X  # this is our input data, of shape (32, 21, 16)
# we will feed it to our model in sequences of length 10

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(X[:, :10, :], np.reshape(X[:, 10, :], (32, 16)))

# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(X[:, 10:20, :], np.reshape(X[:, 20, :], (32, 16)))

# let's reset the states of the LSTM layer:
model.reset_states()

# another way to do it in this case:
model.layers[0].reset_states()
```
注意，`predict`，`fit`，`train_on_batch`，`predict_classes`等方法都会更新模型中状态层的状态。这使得你不但可以进行状态网络的训练，也可以进行状态网络的预测。

##2.8 如何“冻结”网络的层？


“冻结”一个层指的是该层将不参加网络训练，即该层的权重永不会更新。在进行fine-tune时我们经常会需要这项操作。 在使用固定的embedding层处理文本输入时，也需要这个技术。

可以通过向层的构造函数传递`trainable`参数来指定一个层是不是可训练的，如：

```python
frozen_layer = Dense(32,trainable=False)
```
此外，也可以通过将层对象的`trainable`属性设为`True`或`False`来为已经搭建好的模型设置要冻结的层。 在设置完后，需要运行`compile`来使设置生效，例如：

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # this does NOT update the weights of `layer`
trainable_model.fit(data, labels)  # this updates the weights of `layer`
```

##2.9 如何从Sequential模型中去除一个层？
可以通过调用`.pop()`来去除模型的最后一个层，反复调用n次即可去除模型后面的n个层

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```
#3 一些基本概念
##3.1 张量

张量，或tensor，是本文档会经常出现的一个词汇，在此稍作解释。

使用这个词汇的目的是为了表述统一，张量可以看作是向量、矩阵的自然推广，我们用张量来表示广泛的数据类型。

规模最小的张量是0阶张量，即标量，也就是一个数。

当我们把一些数有序的排列起来，就形成了1阶张量，也就是一个向量

如果我们继续把一组向量有序的排列起来，就形成了2阶张量，也就是一个矩阵

把矩阵摞起来，就是3阶张量，我们可以称为一个立方体，具有3个颜色通道的彩色图片就是一个这样的立方体

把立方体摞起来，好吧这次我们真的没有给它起别名了，就叫4阶张量了，不要去试图想像4阶张量是什么样子，它就是个数学上的概念。

张量的阶数有时候也称为维度，或者轴，轴这个词翻译自英文axis。譬如一个矩阵[[1,2],[3,4]]，是一个2阶张量，有两个维度或轴，沿着第0个轴（为了与python的计数方式一致，本文档维度和轴从0算起）你看到的是[1,2]，[3,4]两个向量，沿着第1个轴你看到的是[1,3]，[2,4]两个向量。

要理解“沿着某个轴”是什么意思，不妨试着运行一下下面的代码：

```python
import numpy as np

a = np.array([[1,2],[3,4]])
sum0 = np.sum(a, axis=0)
sum1 = np.sum(a, axis=1)

print sum0
print sum1
```

##3.2 data_format

这是一个无可奈何的问题，在如何表示一组彩色图片的问题上，Theano和TensorFlow发生了分歧，'th'模式，也即Theano模式会把100张RGB三通道的16×32（高为16宽为32）彩色图表示为下面这种形式（100,3,16,32），Caffe采取的也是这种方式。第0个维度是样本维，代表样本的数目，第1个维度是通道维，代表颜色通道数。后面两个就是高和宽了。这种theano风格的数据组织方法，称为“channels_first”，即通道维靠前。

而TensorFlow，的表达形式是（100,16,32,3），即把通道维放在了最后，这种数据组织方式称为“channels_last”。

Keras默认的数据组织形式在~/.keras/keras.json中规定，可查看该文件的image_data_format一项查看，也可在代码中通过K.image_data_format()函数返回，请在网络的训练和测试中保持维度顺序一致。

唉，真是蛋疼，你们商量好不行吗？


##3.3 函数式模型


函数式模型算是本文档比较原创的词汇了，所以这里要说一下

在Keras 0.x中，模型其实有两种，一种叫Sequential，称为序贯模型，也就是单输入单输出，一条路通到底，层与层之间只有相邻关系，跨层连接统统没有。这种模型编译速度快，操作上也比较简单。第二种模型称为Graph，即图模型，这个模型支持多输入多输出，层与层之间想怎么连怎么连，但是编译速度慢。可以看到，Sequential其实是Graph的一个特殊情况。

在Keras1和Keras2中，图模型被移除，而增加了了“functional model API”，这个东西，更加强调了Sequential是特殊情况这一点。一般的模型就称为Model，然后如果你要用简单的Sequential，OK，那还有一个快捷方式Sequential。

由于functional model API在使用时利用的是“函数式编程”的风格，我们这里将其译为函数式模型。总而言之，只要这个东西接收一个或一些张量作为输入，然后输出的也是一个或一些张量，那不管它是什么鬼，统统都称作“模型”。

##3.4 batch
这个概念与Keras无关，老实讲不应该出现在这里的，但是因为它频繁出现，而且不了解这个技术的话看函数说明会很头痛，这里还是简单说一下。

深度学习的优化算法，说白了就是梯度下降。每次的参数更新有两种方式。

第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为Batch gradient descent，批梯度下降。

另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，stochastic gradient descent。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。

为了克服两种方法的缺点，现在一般采用的是一种折中手段，mini-batch gradient decent，小批的梯度下降，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。

基本上现在的梯度下降都是基于mini-batch的，所以Keras的模块中经常会出现batch_size，就是指这个。

顺便说一句，Keras中用的优化器SGD是stochastic gradient descent的缩写，但不代表是一个样本就更新一回，还是基于mini-batch的。

##3.5 epochs

真的不是很想解释这个词，但是新手问的还挺多的…… 简单说，epochs指的就是训练过程中，所有数据将被训练多少次，就这样。

##3.6 shuffle和validation_split的顺序

模型的fit函数有两个参数，shuffle用于将数据打乱，validation_split用于在没有提供验证集的时候，按一定比例从训练集中取出一部分作为验证集

这里有个陷阱是，程序是先执行validation_split，再执行shuffle的，所以会出现这种情况：

假如你的训练集是有序的，比方说正样本在前负样本在后，又设置了validation_split，那么你的验证集中很可能将全部是负样本

同样的，这个东西不会有任何错误报出来，因为Keras不可能知道你的数据有没有经过shuffle，保险起见如果你的数据是没shuffle过的，最好手动shuffle一下


#4 快速开始函数式（Functional）模型

##4.1 第一个模型：全连接网络

`Sequential`当然是实现全连接网络的最好方式，但我们从简单的全连接网络开始，有助于我们学习这部分的内容。在开始前，有几个概念需要澄清：

层对象接受张量为参数，返回一个张量。
输入是张量，输出也是张量的一个框架就是一个模型，通过`Model`定义。
这样的模型可以被像Keras的`Sequential`一样被训练

```python
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training
```

##4.2 所有的模型都是可调用的，就像层一样

利用函数式模型的接口，我们可以很容易的重用已经训练好的模型：你可以把模型当作一个层一样，通过提供一个tensor来调用它。注意当你调用一个模型时，你不仅仅重用了它的结构，也重用了它的权重。

```python
x = Input(shape=(784,))
# This works, and returns the 10-way softmax we defined above.
y = model(x)
```
这种方式可以允许你快速的创建能处理序列信号的模型，你可以很快将一个图像分类的模型变为一个对视频分类的模型，只需要一行代码：

```python
from keras.layers import TimeDistributed

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)
```

##4.3 多输入和多输出模型

使用函数式模型的一个典型场景是搭建多输入、多输出的模型。

考虑这样一个模型。我们希望预测Twitter上一条新闻会被转发和点赞多少次。模型的主要输入是新闻本身，也就是一个词语的序列。但我们还可以拥有额外的输入，如新闻发布的日期等。这个模型的损失函数将由两部分组成，辅助的损失函数评估仅仅基于新闻本身做出预测的情况，主损失函数评估基于新闻和额外信息的预测的情况，即使来自主损失函数的梯度发生弥散，来自辅助损失函数的信息也能够训练Embeddding和LSTM层。在模型中早点使用主要的损失函数是对于深度网络的一个良好的正则方法。总而言之，该模型框图如下：

![multi-input]( /Users/double/Desktop/keras/multi-input.png )

让我们用函数式模型来实现这个框图

主要的输入接收新闻本身，即一个整数的序列（每个整数编码了一个词）。这些整数位于1到10，000之间（即我们的字典有10，000个词）。这个序列有100个单词。

```python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)
```
然后，我们插入一个额外的损失，使得即使在主损失很高的情况下，LSTM和Embedding层也可以平滑的训练。

```python
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
```
再然后，我们将LSTM与额外的输入数据串联起来组成输入，送入模型中：

```python
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
```
最后，我们定义整个2输入，2输出的模型：

```python
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
```
模型定义完毕，下一步编译模型。我们给额外的损失赋0.2的权重。我们可以通过关键字参数`loss_weights`或`loss`来为不同的输出设置不同的损失函数或权值。这两个参数均可为Python的列表或字典。这里我们给`loss`传递单个损失函数，这个损失函数会被应用于所有输出上。

```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
```
编译完成后，我们通过传递训练数据和目标值训练该模型：

```python
model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)
```
因为我们输入和输出是被命名过的（在定义时传递了“name”参数），我们也可以用下面的方式编译和训练模型：

```python
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
```

##4.4 共享层

另一个使用函数式模型的场合是使用共享层的时候。

考虑微博数据，我们希望建立模型来判别两条微博是否是来自同一个用户，这个需求同样可以用来判断一个用户的两条微博的相似性。

一种实现方式是，我们建立一个模型，它分别将两条微博的数据映射到两个特征向量上，然后将特征向量串联并加一个logistic回归层，输出它们来自同一个用户的概率。这种模型的训练数据是一对对的微博。

因为这个问题是对称的，所以处理第一条微博的模型当然也能重用于处理第二条微博。所以这里我们使用一个共享的LSTM层来进行映射。

首先，我们将微博的数据转为（140，256）的矩阵，即每条微博有140个字符，每个单词的特征由一个256维的词向量表示，向量的每个元素为1表示某个字符出现，为0表示不出现，这是一个one-hot编码。

之所以是（140，256）是因为一条微博最多有140个字符，而扩展的ASCII码表编码了常见的256个字符。原文中此处为Tweet，所以对外国人而言这是合理的。如果考虑中文字符，那一个单词的词向量就不止256了。

```python
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))
```
若要对不同的输入共享同一层，就初始化该层一次，然后多次调用它

```python
# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# We can then concatenate the two vectors:
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
```
先暂停一下，看看共享层到底输出了什么，它的输出数据shape又是什么

##4.5 层“节点”的概念

无论何时，当你在某个输入上调用层时，你就创建了一个新的张量（即该层的输出），同时你也在为这个层增加一个“（计算）节点”。这个节点将输入张量映射为输出张量。当你多次调用该层时，这个层就有了多个节点，其下标分别为0，1，2...

在上一版本的Keras中，你可以通过`layer.get_output()`方法来获得层的输出张量，或者通过`layer.output_shape`获得其输出张量的shape。这个版本的Keras你仍然可以这么做（除了`layer.get_output()`被`output`替换）。但如果一个层与多个输入相连，会出现什么情况呢？

如果层只与一个输入相连，那没有任何困惑的地方。`.output`将会返回该层唯一的输出

```python
a = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a
```
但当层与多个输入相连时，会出现问题

```python
a = Input(shape=(140, 256))
b = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

lstm.output
```
上面这段代码会报错

```python
>> AssertionError: Layer lstm_1 has multiple inbound nodes,
hence the notion of "layer output" is ill-defined.
Use `get_output_at(node_index)` instead.
```
通过下面这种调用方式即可解决

```python
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```
对于`input_shape`和`output_shape`也是一样，如果一个层只有一个节点，或所有的节点都有相同的输入或输出shape，那么`input_shape`和`output_shape`都是没有歧义的，并也只返回一个值。但是，例如你把一个相同的`Conv2D`应用于一个大小为(32,32,3)的数据，然后又将其应用于一个(64,64,3)的数据，那么此时该层就具有了多个输入和输出的shape，你就需要显式的指定节点的下标，来表明你想取的是哪个了

```python
a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# Only one input so far, the following will work:
assert conv.input_shape == (None, 32, 32, 3)

conved_b = conv(b)
# now the `.input_shape` property wouldn't work, but this does:
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
```

##4.6 更多的例子

代码示例依然是学习的最佳方式，这里是更多的例子

### inception模型

inception的详细结构参见Google的这篇论文：Going Deeper with Convolutions

```python
from keras.layers import Conv2D, MaxPooling2D, Input

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
```

###卷积层的残差连接

残差网络（Residual Network）的详细信息请参考这篇文章：Deep Residual Learning for Image Recognition

```python
from keras.layers import Conv2D, Input

# input tensor for a 3-channel 256x256 image
x = Input(shape=(256, 256, 3))
# 3x3 conv with 3 output channels (same as input channels)
y = Conv2D(3, (3, 3), padding='same')(x)
# this returns x + y.
z = keras.layers.add([x, y])
```

###共享视觉模型

该模型在两个输入上重用了图像处理的模型，用来判别两个MNIST数字是否是相同的数字

```python
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

# First, define the vision modules
digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# Then define the tell-digits-apart model
digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

# The vision model will be shared, weights and all
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)
```

###视觉问答模型

在针对一幅图片使用自然语言进行提问时，该模型能够提供关于该图片的一个单词的答案

这个模型将自然语言的问题和图片分别映射为特征向量，将二者合并后训练一个logistic回归层，从一系列可能的回答中挑选一个。

```python
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# Now let's get a tensor with the output of our vision model:
image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# Let's concatenate the question vector and the image vector:
merged = keras.layers.concatenate([encoded_question, encoded_image])

# And let's train a logistic regression over 1000 words on top:
output = Dense(1000, activation='softmax')(merged)

# This is our final model:
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# The next stage would be training this model on actual data.
```

###视频问答模型

在做完图片问答模型后，我们可以快速将其转为视频问答的模型。在适当的训练下，你可以为模型提供一个短视频（如100帧）然后向模型提问一个关于该视频的问题，如“what sport is the boy playing？”->“football”

```python
from keras.layers import TimeDistributed

video_input = Input(shape=(100, 224, 224, 3))
# This is our video encoded via the previously trained vision_model (weights are reused)
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector

# This is a model-level representation of the question encoder, reusing the same weights as before:
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# Let's use it to encode the question:
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# And this is our video question answering model:
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)
```


#5 快速开始序贯（Sequential）模型

序贯模型是多个网络层的线性堆叠，也就是“一条路走到黑”。

可以通过向`Sequential`模型传递一个layer的list来构造该模型：

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
Dense(32, units=784),
Activation('relu'),
Dense(10),
Activation('softmax'),
])
```
也可以通过.add()方法一个个的将layer加入模型中：

```python
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
```

##5.1 指定输入数据的shape

模型需要知道输入数据的shape，因此，Sequential的第一层需要接受一个关于输入数据shape的参数，后面的各个层则可以自动的推导出中间数据的shape，因此不需要为每个层都指定这个参数。有几种方法来为第一层指定输入数据的shape

* 传递一个`input_shape的`关键字参数给第一层，`input_shape`是一个tuple类型的数据，其中也可以填入`None`，如果填入`None`则表示此位置可能是任何正整数。数据的batch大小不应包含在其中。
* 有些2D层，如`Dense`，支持通过指定其输入维度`input_dim`来隐含的指定输入数据shape,是一个Int类型的数据。一些3D的时域层支持通过参数`input_dim`和`input_length`来指定输入shape。
* 如果你需要为输入指定一个固定大小的batch_size（常用于stateful RNN网络），可以传递`batch_size`参数到一个层中，例如你想指定输入张量的batch大小是32，数据shape是（6，8），则你需要传递`batch_size=32`和`input_shape=(6,8)`。

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
```
```python
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
```

##5.2 编译

在训练模型之前，我们需要通过`compile`来对学习过程进行配置。`compile`接收三个参数：

* 优化器`optimizer`：该参数可指定为已预定义的优化器名，如`rmsprop`、`adagrad`，或一个`Optimizer`类的对象，详情见optimizers
* 损失函数`loss`：该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如`categorical_crossentropy`、`mse`，也可以为一个损失函数。详情见losses
* 指标列表`metrics`：对分类问题，我们一般将该列表设置为metrics=['accuracy']。指标可以是一个预定义指标的名字,也可以是一个用户定制的函数.指标函数应该返回单个张量,或一个完成`metric_name - > metric_value`映射的字典.

```python
# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```

##5.3 训练

Keras以Numpy数组作为输入数据和标签的数据类型。训练模型一般使用`fit`函数，该函数的详情见这里。下面是一些例子。

```python
# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)

# For a single-input model with 10 classes (categorical classification):
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

##5.4 例子

这里是一些帮助你开始的例子

在Keras代码包的examples文件夹中，你将找到使用真实数据的示例模型：

* CIFAR10 小图片分类：使用CNN和实时数据提升
* IMDB 电影评论观点分类：使用LSTM处理成序列的词语
* Reuters（路透社）新闻主题分类：使用多层感知器（MLP）
* MNIST手写数字识别：使用多层感知器和CNN
* 字符级文本生成：使用LSTM ...

###基于多层感知器的softmax多分类：


```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

###MLP的二分类：
MLP（Multi-Layer Perceptron）,多层感知器

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

###类似VGG的卷积神经网络：

```python
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# Generate dummy data
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) #Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
```

###使用LSTM的序列分类

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

###使用1D卷积的序列分类

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

###用于序列分类的栈式LSTM

在该模型中，我们将三个LSTM堆叠在一起，是该模型能够学习更高层次的时域特征表示。

开始的两层LSTM返回其全部输出序列，而第三层LSTM只返回其输出序列的最后一步结果，从而其时域维度降低（即将输入序列转换为单个向量）

![](/Users/double/Desktop/keras/regular_stacked_lstm.png
)

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
```

###采用stateful LSTM的相同模型
stateful LSTM的特点是，在处理过一个batch的训练数据后，其内部状态（记忆）会被作为下一个batch的训练数据的初始状态。状态LSTM使得我们可以在合理的计算复杂度内处理较长序列

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# Generate dummy validation data
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))
```


#6 关于Keras的“层”（Layer）

所有的Keras层对象都有如下方法：

* `layer.get_weights()`：返回层的权重（numpy array）
* `layer.set_weights(weights)`：从numpy array中将权重加载到该层中，要求numpy array的形状与* `layer.get_weights()`的形状相同
* `layer.get_config()`：返回当前层配置信息的字典，层也可以借由配置信息重构:

```python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```
或者：

```python
from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})
                            
```
如果层仅有一个计算节点（即该层不是共享层），则可以通过下列方法获得输入张量、输出张量、输入数据的形状和输出数据的形状：

* `layer.input`
* `layer.output`
* `layer.input_shape`
* `layer.output_shape`
如果该层有多个计算节点（参考层计算节点和共享层）。可以使用下面的方法

* `layer.get_input_at(node_index)`
* `layer.get_output_at(node_index)`
* `layer.get_input_shape_at(node_index)`
* `layer.get_output_shape_at(node_index)`

#7 卷积层

##7.1 Conv1D层

```python
keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
一维卷积层（即时域卷积），用以在一维输入信号上进行邻域滤波。当使用该层作为首层时，需要提供关键字参数`input_shape`。例如`(10,128)`代表一个长为10的序列，序列中每个信号为128向量。而`(None, 128)`代表变长的128维向量序列。

该层生成将输入信号与卷积核按照单一的空域（或时域）方向进行卷积。如果`use_bias=True`，则还会加上一个偏置项，若`activation`不为None，则输出为经过激活函数的输出。

###参数
* filters：卷积核的数目（即输出的维度）
* kernel_size：整数或由单个整数构成的list/tuple，卷积核的空域或时域窗长度
* strides：整数或由单个整数构成的list/tuple，为卷积的步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
* padding：补0策略，为“valid”, “same” 或“causal”，“causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。参考WaveNet: A Generative Model for Raw Audio, section 2.1.。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
* activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
* dilation _ rate：整数或由单个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
* use_bias:布尔值，是否使用偏置项
* kernel _ initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializersbias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* kernel_regularizer：施加在权重上的正则项，为Regularizer对象
* bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
* activity_regularizer：施加在输出上的正则项，为Regularizer对象
* kernel_constraints：施加在权重上的约束项，为Constraints对象
* bias_constraints：施加在偏置上的约束项，为Constraints对象

###输入shape

形如（samples，steps，input_dim）的3D张量

###输出shape

形如（samples，new_ steps，nb_ filter）的3D张量，因为有向量填充的原因，steps的值会改变

【Tips】可以将Convolution1D看作Convolution2D的快捷版，对例子中（10，32）的信号进行1D卷积相当于对其进行卷积核为（filter_length, 32）的2D卷积。【@3rduncle】

##7.2 Conv2D层

```python
keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
二维卷积层，即对图像的空域卷积。该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供`input_shape`参数。例如`input_shape = (128,128,3)`代表128*128的彩色RGB图像`（data_format='channels_last'）`

###参数
* filters：卷积核的数目（即输出的维度）
* kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
* strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
* padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
* activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
* dilation_ rate：单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
* data_ format：字符串，“channels_ first”或“channels_ last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_ dim_ ordering，“channels_ last”对应原本的“tf”，“channels_ first”对应原本的“th”。以128x128的RGB图像为例，“channels_ first”应将数据组织为（3,128,128），而“channels_ last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_ last”。
* use_bias:布尔值，是否使用偏置项
* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* kernel_regularizer：施加在权重上的正则项，为Regularizer对象
* bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
* activity_regularizer：施加在输出上的正则项，为Regularizer对象
* kernel_constraints：施加在权重上的约束项，为Constraints对象
* bias_constraints：施加在偏置上的约束项，为Constraints对象

###输入shape
‘channels_ first’模式下，输入形如（samples,channels，rows，cols）的4D张量

‘channels_ last’模式下，输入形如（samples，rows，cols，channels）的4D张量

注意这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的`input_shape`，请参考下面提供的例子。

###输出shape
‘channels_ first’模式下，为形如（samples，nb_filter, new_rows, new_cols）的4D张量

‘channels_ last’模式下，为形如（samples，new_rows, new_cols，nb_filter）的4D张量

输出的行列数可能会因为填充方法而改变

##7.3 SeparableConv2D层

##7.4 Conv2DTranspose层

##7.5 Conv3D层

##7.6 Cropping1D层

##7.7 Cropping2D层

##7.8 Cropping3D层

##7.9 UpSampling1D层

##7.10 UpSampling2D层

##7.11 UpSampling3D层

##7.12 ZeroPadding1D层

##7.13 ZeroPadding2D层

##7.14 ZeroPadding3D层


#8 常用层

常用层对应于core模块，core内部定义了一系列常用的网络层，包括全连接、激活层等

##8.1 Dense层
```python
keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
Dense就是常用的全连接层，所实现的运算是`output = activation(dot(input, kernel)+bias)`。其中`activation`是逐元素计算的激活函数，`kernel`是本层的权值矩阵，`bias`为偏置向量，只有当`use_bias=True`才会添加。

如果本层的输入数据的维度大于2，则会先被压为与`kernel`相匹配的大小。

这里是一个使用示例：

```python
# as first layer in a sequential model:
# as first layer in a sequential model:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(32))
```
##参数：
* units：大于0的整数，代表该层的输出维度。
* activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）
* Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
* use_bias: 布尔值，是否使用偏置项
* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* bias_initializer：偏置向量初始化方法，为预定义初始化方法名的字符串，或用于初始化偏置向量的初始化器。参考initializers
* kernel_regularizer：施加在权重上的正则项，为Regularizer对象
* bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
* activity_regularizer：施加在输出上的正则项，为Regularizer对象
* kernel_constraints：施加在权重上的约束项，为Constraints对象
* bias_constraints：施加在偏置上的约束项，为Constraints对象

###输入

形如(batch_ size, ..., input_dim)的nD张量，最常见的情况为(batch_size, input_dim)的2D张量

###输出

形如(batch_ size, ..., units)的nD张量，最常见的情况为(batch_size, units)的2D张量


##8.2 Activation层

```python
keras.layers.core.Activation(activation)
```
激活层对一个层的输出施加激活函数

##8.3 Dropout层

```python
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
```
为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。

参数

* rate：0~1的浮点数，控制需要断开的神经元的比例
* noise_ shape：整数张量，为将要应用在输入上的二值Dropout mask的shape，例如你的输入为(batch_ size, timesteps, features)，并且你希望在各个时间步上的Dropout mask都相同，则可传入noise_ shape=(batch_ size, 1, features)。
* seed：整数，使用的随机数种子

##8.4 Flatten层

```python
keras.layers.core.Flatten()
```
Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。

###例子

```python
model = Sequential()
model.add(Convolution2D(64, 3, 3,
            border_mode='same',
            input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
```

##8.5 Reshape层

```python
keras.layers.core.Reshape(target_shape)
```
Reshape层用来将输入shape转换为特定的shape

###参数
* target_shape：目标shape，为整数的tuple，不包含样本数目的维度（batch大小）

###输入shape
任意，但输入的shape必须固定。当使用该层为模型首层时，需要指定input_shape参数

###输出shape
`(batch_size,)+target_shape`

###例子

```python
# as first layer in a Sequential model
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# note: `None` is the batch dimension

# as intermediate layer in a Sequential model
model.add(Reshape((6, 2)))
# now: model.output_shape == (None, 6, 2)

# also supports shape inference using `-1` as dimension
model.add(Reshape((-1, 2, 2)))
# now: model.output_shape == (None, 3, 2, 2)
```

##8.6 Permute层

```python
keras.layers.core.Permute(dims)
```
Permute层将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层。

###参数
dims：整数tuple，指定重排的模式，不包含样本数的维度。重拍模式的下标从1开始。例如（2，1）代表将输入的第二个维度重拍到输出的第一个维度，而将输入的第一个维度重排到第二个维度
###例子

```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# now: model.output_shape == (None, 64, 10)
# note: `None` is the batch dimension
```
###输入shape
任意，当使用激活层作为第一层时，要指定input_shape

###输出shape
与输入相同，但是其维度按照指定的模式重新排列

##8.7 RepeatVector层

```python
keras.layers.core.RepeatVector(n)
```
RepeatVector层将输入重复n次

###参数
n：整数，重复的次数

###输入shape
形如（nb_samples, features）的2D张量

###输出shape
形如（nb_samples, n, features）的3D张量

###例子

```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# now: model.output_shape == (None, 32)
# note: `None` is the batch dimension

model.add(RepeatVector(3))
# now: model.output_shape == (None, 3, 32)
```

##8.8 Lambda层

```python
keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)
```
本函数用以对上一层的输出施以任何Theano/TensorFlow表达式

###参数
* function：要实现的函数，该函数仅接受一个变量，即上一层的输出
* output_shape：函数应该返回的值的shape，可以是一个tuple，也可以是一个根据输入shape计算输出shape的函数
* mask: 掩膜
* arguments：可选，字典，用来记录向函数中传递的其他关键字参数

###例子

```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)

model.add(Lambda(antirectifier,
         output_shape=antirectifier_output_shape))

```

###输入shape
任意，当使用该层作为第一层时，要指定input_shape

###输出shape
由output_shape参数指定的输出shape，当使用tensorflow时可自动推断

##8.9 ActivityRegularizer层

```python
keras.layers.core.ActivityRegularization(l1=0.0, l2=0.0)
```
经过本层的数据不会有任何变化，但会基于其激活值更新损失函数值

###参数
* l1：1范数正则因子（正浮点数）
* l2：2范数正则因子（正浮点数）

###输入shape
任意，当使用该层作为第一层时，要指定input_shape

###输出shape
与输入shape相同

##8.10 Masking层

```python
keras.layers.core.Masking(mask_value=0.0)
```
使用给定的值对输入的序列信号进行“屏蔽”，用以定位需要跳过的时间步

对于输入张量的时间步，即输入张量的第1维度（维度从0开始算，见例子），如果输入张量在该时间步上都等于`mask_value`，则该时间步将在模型接下来的所有层（只要支持masking）被跳过（屏蔽）。

如果模型接下来的一些层不支持masking，却接受到masking过的数据，则抛出异常。

###例子

考虑输入数据`x`是一个形如(samples,timesteps,features)的张量，现将其送入LSTM层。因为你缺少时间步为3和5的信号，所以你希望将其掩盖。这时候应该：

赋值`x[:,3,:] = 0.，x[:,5,:] = 0.`
在LSTM层之前插入`mask_value=0.`的`Masking`层

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```

#9 嵌入层 Embedding
###Embedding层

```python
keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```
嵌入层将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]
Embedding层只能作为模型的第一层

###参数
* input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1
* output_dim：大于0的整数，代表全连接嵌入的维度
* embeddings_initializer: 嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* embeddings_regularizer: 嵌入矩阵的正则项，为Regularizer对象
* embeddings_constraint: 嵌入矩阵的约束项，为Constraints对象
* mask_ zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用递归层处理变长输入时有用。设置为True的话，模型中后续的层必须都支持masking，否则会抛出异常。如果该值为True，则下标0在字典中不可用，input_ dim应设置为|vocabulary| + 1。
* input_ length：当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。

###输入shape
形如（samples，sequence_ length）的2D张量

###输出shape
形如(samples, sequence_ length, output_dim)的3D张量

###例子

```python
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

#10 局部连接层LocallyConnceted

##10.1 LocallyConnected1D层

##10.2 LocallyConnected2D层

#11 Merge层

Merge层提供了一系列用于融合两个层或两个张量的层对象和方法。以大写首字母开头的是Layer类，以小写字母开头的是张量的函数。小写字母开头的张量函数在内部实际上是调用了大写字母开头的层。

##11.1 Add

```python
keras.layers.Add()
```
添加输入列表的图层。
该层接收一个相同shape列表张量，并返回它们的和，shape不变。

###Example

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

##11.2 SubStract

```python
keras.layers.Subtract()
```
两个输入的层相减。
它将大小至少为2，相同Shape的列表张量作为输入，并返回一个张量（输入[0] - 输入[1]），也是相同的Shape。

###Example

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# Equivalent to subtracted = keras.layers.subtract([x1, x2])
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

##11.3 其它
###Multiply
```python
keras.layers.Multiply()
```
该层接收一个列表的同shape张量，并返回它们的逐元素积的张量，shape不变。

###Average
```python
keras.layers.Average()
```
该层接收一个列表的同shape张量，并返回它们的逐元素均值，shape不变。

###Maximum
```python
keras.layers.Maximum()
```
该层接收一个列表的同shape张量，并返回它们的逐元素最大值，shape不变。

###Concatenate
```python
keras.layers.Concatenate(axis=-1)
```
该层接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量。

####参数
* axis: 想接的轴
* **kwargs: 普通的Layer关键字参数

###Dot
```python
keras.layers.Dot(axes, normalize=False)
```
计算两个tensor中样本的张量乘积。例如，如果两个张量a和b的shape都为（batch_ size, n），则输出为形如（batch_ size,1）的张量，结果张量每个batch的数据都是a[i,:]和b[i,:]的矩阵（向量）点积。

####参数

* axes: 整数或整数的tuple，执行乘法的轴。
n* ormalize: 布尔值，是否沿执行成绩的轴做L2规范化，如果设为True，那么乘积的输出是两个样本的余弦相似性。
* **kwargs: 普通的Layer关键字参数

##11.4 张量的函数（小写）
### add
```python
keras.layers.add(inputs)
```
Add层的函数式包装

####参数：
* inputs: 长度至少为2的张量列表A
* **kwargs: 普通的Layer关键字参数

####返回值
输入列表张量之和

####Example
```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

###subtract
```python
keras.layers.subtract(inputs)
```
Subtract层的函数式包装

####参数：
* inputs: 长度至少为2的张量列表A
* **kwargs: 普通的Layer关键字参数

####返回值
输入张量列表的差别

####Example
```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
subtracted = keras.layers.subtract([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

###multiply
```python
keras.layers.multiply(inputs)
```
Multiply的函数式包装

####参数：
* inputs: 长度至少为2的张量列表
* **kwargs: 普通的Layer关键字参数

####返回值
输入列表张量之逐元素积

###average
```python
keras.layers.average(inputs)
```
Average的函数包装

####参数：
* inputs: 长度至少为2的张量列表
* **kwargs: 普通的Layer关键字参数

####返回值
输入列表张量之逐元素均值

###maximum
```python
keras.layers.maximum(inputs)
```
Maximum的函数包装
####参数：
* inputs: 长度至少为2的张量列表
* **kwargs: 普通的Layer关键字参数

####返回值
输入列表张量之逐元素均值

###concatenate
```python
keras.layers.concatenate(inputs, axis=-1)
```
Concatenate的函数包装

####参数
* inputs: 长度至少为2的张量列
* axis: 相接的轴
* **kwargs: 普通的Layer关键字参数

###dot
```pyhton
keras.layers.dot(inputs, axes, normalize=False)
```
Dot的函数包装
####参数
* inputs: 长度至少为2的张量列
* axes: 整数或整数的tuple，执行乘法的轴。
* normalize: 布尔值，是否沿执行成绩的轴做L2规范化，如果设为True，那么乘积的输出是两个样本的余弦相似性。
* **kwargs: 普通的Layer关键字参数

#12噪声层Noise

##12.1 GaussianNoise层
```python
keras.layers.noise.GaussianNoise(stddev)
```
为数据施加0均值，标准差为stddev的加性高斯噪声。该层在克服过拟合时比较有用，你可以将它看作是随机的数据提升。高斯噪声是需要对输入数据进行破坏时的自然选择。

因为这是一个起正则化作用的层，该层只在训练时才有效。

###参数
* stddev：浮点数，代表要产生的高斯噪声标准差

###输入shape
任意，当使用该层为模型首层时需指定input_shape参数

###输出shape
与输入相同

##12.2 GaussianDropout层
```python
keras.layers.noise.GaussianDropout(rate)
```
为层的输入施加以1为均值，标准差为sqrt(rate/(1-rate)的乘性高斯噪声
因为这是一个起正则化作用的层，该层只在训练时才有效。

###参数
* rate：浮点数，断连概率，与Dropout层相同

###输入shape
任意，当使用该层为模型首层时需指定input_shape参数
###输出shape
与输入相同

##12.3 AlphaDropout
```python
keras.layers.noise.AlphaDropout(rate, noise_shape=None, seed=None)
```
对输入施加Alpha Dropout
Alpha Dropout是一种保持输入均值和方差不变的Dropout，该层的作用是即使在dropout时也保持数据的自规范性。 通过随机对负的饱和值进行激活，Alphe Drpout与selu激活函数配合较好。

###参数
* rate: 浮点数，类似Dropout的Drop比例。乘性mask的标准差将保证为sqrt(rate / (1 - rate)).
* seed: 随机数种子

###输入shape
任意，当使用该层为模型首层时需指定input_shape参数
###输出shape
与输入相同

#13 （批）规范化BatchNormalization

##BatchNormalization层
```python
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```
该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1

###参数
* axis: 整数，指定要规范化的轴，通常为特征轴。例如在进行data_format="channels_first的2D卷积后，一般会设axis=1。
* momentum: 动态均值的动量
* epsilon：大于0的小浮点数，用于防止除0错误
* center: 若设为True，将会将beta作为偏置加上去，否则忽略参数beta
* scale: 若设为True，则会乘以gamma，否则不使用gamma。当下一层是线性的时，可以设False，因为scaling的操作将被下一层执行。
* beta_initializer：beta权重的初始方法
* gamma_initializer: gamma的初始化方法
* moving_mean_initializer: 动态均值的初始化方法
* moving_variance_initializer: 动态方差的初始化方法
* beta_regularizer: 可选的beta正则
* gamma_regularizer: 可选的gamma正则
* beta_constraint: 可选的beta约束
* gamma_constraint: 可选的gamma约束

###输入shape
任意，当使用本层为模型首层时，指定input_shape参数时有意义。
###输出shape
与输入shape相同
###【Tips】BN层的作用
（1）加速收敛 （2）控制过拟合，可以少用或不用Dropout和正则 （3）降低网络对初始化权重不敏感 （4）允许使用较大的学习率

#14 池化层

##14.1 MaxPooling1D层
```python
keras.layers.pooling.MaxPooling1D(pool_size=2, strides=None, padding='valid')
```
对时域1D信号进行最大值池化
###参数
* pool_ size：整数，池化窗口大小
* strides：整数或None，下采样因子，例如设2将会使得输出shape为输入的一半，若为None则默认值为pool_ size。
* padding：‘valid’或者‘same’

###输入shape
形如（samples，steps，features）的3D张量
###输出shape
形如（samples，downsampled_steps，features）的3D张量
##14.2 MaxPooling2D层

##14.3 MaxPooling3D层

##14.4 AveragePooling1D层

##14.5 AveragePooling2D层

##14.6 AveragePooling3D层

##14.7 GlobalMaxPooling1D层

##14.8 GlobalAveragePooling1D层

##14.9 GlobalMaxPooling2D层

##14.10 GlobalMaxPooling2D层


#15 循环层Recurrent

##15.1 Recurrent层
```python
keras.layers.recurrent.Recurrent(return_sequences=False, go_backwards=False, stateful=False, unroll=False, implementation=0)
```
这是循环层的抽象类，请不要在模型中直接应用该层（因为它是抽象类，无法实例化任何对象）。请使用它的子类`LSTM`，`GRU`或`SimpleRNN`。

所有的循环层（`LSTM,GRU,SimpleRNN`）都继承本层，因此下面的参数可以在任何循环层中使用。

###参数
* weights：numpy array的list，用以初始化权重。该list形如`[(input_dim, output_dim),(output_dim, output_dim),(output_dim,)]`
* return_sequences：布尔值，默认`False`，控制返回类型。若为`True`则返回整个序列，否则仅返回输出序列的最后一个输出
* go_ backwards：布尔值，默认为`False`，若为`True`，则逆向处理输入序列并返回逆序后的序列
* stateful：布尔值，默认为`False`，若为`True`，则一个batch中下标为i的样本的最终状态将会用作下一个batch同样下标的样本的初始状态。
* unroll：布尔值，默认为False，若为True，则循环层将被展开，否则就使用符号化的循环。当使用TensorFlow为后端时，循环网络本来就是展开的，因此该层不做任何事情。层展开会占用更多的内存，但会加速RNN的运算。层展开只适用于短序列。
* implementation：0，1或2， 若为0，则RNN将以更少但是更大的矩阵乘法实现，因此在CPU上运行更快，但消耗更多的内存。如果设为1，则RNN将以更多但更小的矩阵乘法实现，因此在CPU上运行更慢，在GPU上运行更快，并且消耗更少的内存。如果设为2（仅LSTM和GRU可以设为2），则RNN将把输入门、遗忘门和输出门合并为单个矩阵，以获得更加在GPU上更加高效的实现。注意，RNN dropout必须在所有门上共享，并导致正则效果性能微弱降低。
* input_ dim：输入维度，当使用该层为模型首层时，应指定该值（或等价的指定input_shape)
* input_ length：当输入序列的长度固定时，该参数为输入序列的长度。当需要在该层后连接Flatten层，然后又要连接Dense层时，需要指定该参数，否则全连接的输出无法计算出来。注意，如果循环层不是网络的第一层，你需要在网络的第一层中指定序列的长度（通过input_shape指定）。

###输入shape
形如（samples，timesteps，input_dim）的3D张量
###输出shape
如果`return_sequences=True`：返回形如（samples，timesteps，output_ dim）的3D张量
否则，返回形如（samples，output_dim）的2D张量

###例子
```python
# as the first layer in a Sequential model
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))
# now model.output_shape == (None, 32)
# note: `None` is the batch dimension.

# the following is identical:
model = Sequential()
model.add(LSTM(32, input_dim=64, input_length=10))

# for subsequent layers, no need to specify the input size:
model.add(LSTM(16))

# to stack recurrent layers, you must use return_sequences=True
# on any recurrent layer that feeds into another recurrent layer.
# note that you only need to specify the input size on the first layer.
model = Sequential()
model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(10))
```

###指定RNN初始状态的注意事项
可以通过设置initial_ state用符号式的方式指定RNN层的初始状态。即，initial_ stat的值应该为一个tensor或一个tensor列表，代表RNN层的初始状态。

也可以通过设置reset_states参数用数值的方法设置RNN的初始状态，状态的值应该为numpy数组或numpy数组的列表，代表RNN层的初始状态。

###屏蔽输入数据（Masking）
循环层支持通过时间步变量对输入数据进行Masking，如果想将输入数据的一部分屏蔽掉，请使用Embedding层并将参数`mask_zero`设为`True`。

###使用状态RNN的注意事项
可以将RNN设置为‘stateful’，意味着由每个batch计算出的状态都会被重用于初始化下一个batch的初始状态。状态RNN假设连续的两个batch之中，相同下标的元素有一一映射关系。

要启用状态RNN，请在实例化层对象时指定参数`stateful=True`，并在Sequential模型使用固定大小的batch：通过在模型的第一层传入`batch_size=(...)`和`input_shape`来实现。在函数式模型中，对所有的输入都要指定相同的`batch_size`。

如果要将循环层的状态重置，请调用`.reset_states()`，对模型调用将重置模型中所有状态RNN的状态。对单个层调用则只重置该层的状态。

##15.2 SimpleRNN层
```python
keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```
全连接RNN网络，RNN的输出会被回馈到输入

###参数
* units：输出维度
* activation：激活函数，为预定义的激活函数名（参考激活函数）
* use_ bias: 布尔值，是否使用偏置项
* kernel_ initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* recurrent_ initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* bias_ initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* kernel_ regularizer：施加在权重上的正则项，为Regularizer对象
* bias_ regularizer：施加在偏置向量上的正则项，为Regularizer对象
* recurrent_ regularizer：施加在循环核上的正则项，为Regularizer对象
* activity_ regularizer：施加在输出上的正则项，为Regularizer对象
* kernel_ constraints：施加在权重上的约束项，为Constraints对象
* recurrent_ constraints：施加在循环核上的约束项，为Constraints对象
* bias_ constraints：施加在偏置上的约束项，为Constraints对象
* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
其他参数参考Recurrent的说明


##15.3 GRU层
```python
keras.layers.recurrent.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```
门限循环单元（详见参考文献）

###参数
* units：输出维度
* activation：激活函数，为预定义的激活函数名（参考激活函数）
* use_bias: 布尔值，是否使用偏置项
* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* kernel_regularizer：施加在权重上的正则项，为Regularizer对象
* bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
* recurrent_regularizer：施加在循环核上的正则项，为Regularizer对象
* activity_regularizer：施加在输出上的正则项，为Regularizer对象
* kernel_constraints：施加在权重上的约束项，为Constraints对象
* recurrent_constraints：施加在循环核上的约束项，为Constraints对象
* bias_constraints：施加在偏置上的约束项，为Constraints对象
* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例

其他参数参考Recurrent的说明

##15.4 LSTM层
```python
keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```
Keras长短期记忆模型，关于此算法的详情，请参考本教程

###参数
* units：输出维度
* activation：激活函数，为预定义的激活函数名（参考激活函数）
* recurrent_activation: 为循环步施加的激活函数（参考激活函数）
* use_bias: 布尔值，是否使用偏置项
* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
* kernel_regularizer：施加在权重上的正则项，为Regularizer对象
* bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
* recurrent_regularizer：施加在循环核上的正则项，为Regularizer对象
* activity_regularizer：施加在输出上的正则项，为Regularizer对象
* kernel_constraints：施加在权重上的约束项，为Constraints对象
* recurrent_constraints：施加在循环核上的约束项，为Constraints对象
* bias_constraints：施加在偏置上的约束项，为Constraints对象
* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例

其他参数参考Recurrent的说明

##15.5 ConvLSTM2D层

##15.6 SimpleRNNCell层

##15.7 GRUCell层

##15.8 LSTMCell层

##15.9 StackedRNNCells层

##15.10 CuDNNGRU层

##15.11 CuDNNLSTM层



#16 包装器Wrapper

##16.1 TimeDistributed包装器
```python
keras.layers.wrappers.TimeDistributed(layer)
```
该包装器可以把一个层应用到输入的每一个时间步上

###参数
* layer：Keras层对象

输入至少为3D张量，下标为1的维度将被认为是时间维

例如，考虑一个含有32个样本的batch，每个样本都是10个向量组成的序列，每个向量长为16，则其输入维度为`(32,10,16)`，其不包含batch大小的`input_shape`为`(10,16)`

我们可以使用包装器`TimeDistributed`包装`Dense`，以产生针对各个时间步信号的独立全连接：

```python
# as the first layer in a model
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# now model.output_shape == (None, 10, 8)

# subsequent layers: no need for input_shape
model.add(TimeDistributed(Dense(32)))
# now model.output_shape == (None, 10, 32)
```
程序的输出数据shape为`(32,10,32)`

使用`TimeDistributed`包装`Dense`严格等价于`layers.TimeDistribuedDense`。不同的是包装器`TimeDistribued`还可以对别的层进行包装，如这里对`Convolution2D`包装：

```python
model = Sequential()
model.add(TimeDistributed(Convolution2D(64, 3, 3), input_shape=(10, 3, 299, 299)))
```

##16.2 Bidirectional包装器
```python
keras.layers.wrappers.Bidirectional(layer, merge_mode='concat', weights=None)
```
双向RNN包装器,对序列进行前向和后向计算。

###参数
* layer：Recurrent对象
* merge_mode：前向和后向RNN输出的结合方式，为sum,mul,concat,ave和None之一，若设为None，则返回值不结合，而是以列表的形式返回

###例子
```python
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```


#17 泛型模型(函数式)接口
Keras的泛型模型为Model，即广义的拥有输入和输出的模型，我们使用Model来初始化一个泛型模型

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(input=a, output=b)
```
在这里，我们的模型以a为输入，以b为输出，同样我们可以构造拥有多输入和多输出的模型

```python
model = Model(input=[a1, a2], output=[b1, b3, b3])
```
###常用Model属性
* `model.layers`：组成模型图的各个层
* `model.inputs`：模型的输入张量列表
* `model.outputs`：模型的输出张量列表

##Model模型方法

###compile

```python
compile(self, optimizer, loss, metrics=[], loss_weights=None, sample_weight_mode=None)
```
本函数编译模型以供训练，参数有

* optimizer：优化器，为预定义优化器名或优化器对象，参考优化器
* loss：目标函数，为预定义损失函数名或一个目标函数，参考目标函数
* metrics：列表，包含评估模型在训练和测试时的性能的指标，典型用法是`metrics=['accuracy']`如果要在多输出模型中为不同的输出指定不同的指标，可像该参数传递一个字典，例如`metrics={'ouput_a': 'accuracy'}`
* sample_weight_mode：如果你需要按时间步为样本赋权（2D权矩阵），将该值设为“temporal”。默认为“None”，代表按样本赋权（1D权）。如果模型有多个输出，可以向该参数传入指定sample_weight_mode的字典或列表。在下面fit函数的解释中有相关的参考内容。
* kwargs：使用TensorFlow作为后端请忽略该参数，若使用Theano作为后端，kwargs的值将会传递给 K.function

【Tips】如果你只是载入模型并利用其predict，可以不用进行compile。在Keras中，compile主要完成损失函数和优化器的一些配置，是为训练服务的。predict会在内部进行符号函数的编译工作（通过调用_make_predict_function生成函数）【@白菜，@我是小将】

###fit
```python
fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
```
本函数用以训练模型，参数有：

* x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array。如果模型的每个输入都有名字，则可以传入一个字典，将输入名与其输入数据对应起来。
* y：标签，numpy array。如果模型有多个输出，可以传入一个numpy array的list。如果模型的输出拥有名字，则可以传入一个字典，将输出名与其标签对应起来。
* batch_ size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
* nb_ epoch：整数，训练的轮数，训练数据将会被遍历nb_ epoch次。Keras中nb开头的变量均为"number of"的意思
* verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
* callbacks：list，其中的元素是`keras.callbacks.Callback`的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
* validation_ split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。
* validation_ data：形式为（X，y）或（X，y，sample_ weights）的tuple，是指定的验证集。此参数将覆盖validation_ spilt。
* shuffle：布尔值，表示是否在训练过程中每个epoch前随机打乱输入样本的顺序。
* class_ weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）。该参数在处理非平衡的训练数据（某些类的训练样本数很少）时，可以使得损失函数对样本数不足的数据更加关注。
* sample_ weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了`sample_weight_mode='temporal'`。

`fit`函数返回一个`History`的对象，其`History.history`属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况


###evaluate
```python
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
```
本函数按batch计算在某些输入数据上模型的误差，其参数有：

* x：输入数据，与fit一样，是numpy array或numpy array的list
* y：标签，numpy array
* batch_size：整数，含义同fit的同名参数
* verbose：含义同fit的同名参数，但只能取0或1
* sample_weight：numpy array，含义同fit的同名参数

本函数返回一个测试误差的标量值（如果模型没有其他评价指标），或一个标量的list（如果模型还有其他的评价指标）。`model.metrics_names`将给出list中各个值的含义。

如果没有特殊说明，以下函数的参数均保持与fit的同名参数相同的含义

如果没有特殊说明，以下函数的verbose参数（如果有）均只能取0或1

###predict
```python
predict(self, x, batch_size=32, verbose=0)
```
本函数按batch获得输入数据对应的输出，其参数有：

函数的返回值是预测值的numpy array

###train_ on_ batch
```python
train_on_batch(x, y, class_weight=None, sample_weight=None)
```
本函数在一个batch的数据上进行一次参数更新

函数返回训练误差的标量值或标量值的list，与evaluate的情形相同。

###test_on_batch
```python
test_on_batch(x, y, sample_weight=None)
```
本函数在一个batch的样本上对模型进行评估

函数的返回与evaluate的情形相同

###predict_ on_ batch
```python
predict_on_batch(x)
```
本函数在一个batch的样本上对模型进行测试

函数返回模型在一个batch上的预测结果


###fit_ generator
...
###evaluate_ generator
...
###predict_ generator
...
###get_ layer
...


#18 Sequential模型接口

##Sequential模型方法

###compile
```python
compile(optimizer, loss, metrics=[], sample_weight_mode=None)
```
编译用来配置模型的学习过程，其参数有

optimizer：字符串（预定义优化器名）或优化器对象，参考优化器
loss：字符串（预定义损失函数名）或目标函数，参考目标函数
metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是metrics=['accuracy']
sample_weight_mode：如果你需要按时间步为样本赋权（2D权矩阵），将该值设为“temporal”。默认为“None”，代表按样本赋权（1D权）。在下面fit函数的解释中有相关的参考内容。
kwargs：使用TensorFlow作为后端请忽略该参数，若使用Theano作为后端，kwargs的值将会传递给 K.function

```python
model = Sequential()
model.add(Dense(32, input_shape=(500,)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
      loss='categorical_crossentropy',
      metrics=['accuracy'])
      
```

###fit
```python
fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
```
本函数将模型训练nb_epoch轮，其参数有：

x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
y：标签，numpy array
batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为"number of"的意思
verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。
validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。
class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）
sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。
fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况


###evaluate
```python
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
```
本函数按batch计算在某些输入数据上模型的误差，其参数有：

x：输入数据，与fit一样，是numpy array或numpy array的list
y：标签，numpy array
batch_size：整数，含义同fit的同名参数
verbose：含义同fit的同名参数，但只能取0或1
sample_weight：numpy array，含义同fit的同名参数
本函数返回一个测试误差的标量值（如果模型没有其他评价指标），或一个标量的list（如果模型还有其他的评价指标）。model.metrics_names将给出list中各个值的含义。

如果没有特殊说明，以下函数的参数均保持与fit的同名参数相同的含义

如果没有特殊说明，以下函数的verbose参数（如果有）均只能取0或1

###predict
```python
predict(self, x, batch_size=32, verbose=0)
```
本函数按batch获得输入数据对应的输出，其参数有：

函数的返回值是预测值的numpy array

###predict_classes
```python
predict_classes(self, x, batch_size=32, verbose=1)
```
本函数按batch产生输入数据的类别预测结果

函数的返回值是类别预测结果的numpy array或numpy

###predict_proba
```python
predict_proba(self, x, batch_size=32, verbose=1)
```
本函数按batch产生输入数据属于各个类别的概率

函数的返回值是类别概率的numpy array

###train_on_batch
```python
train_on_batch(self, x, y, class_weight=None, sample_weight=None)
```
本函数在一个batch的数据上进行一次参数更新

函数返回训练误差的标量值或标量值的list，与evaluate的情形相同。

###test_on_batch
```python
test_on_batch(self, x, y, sample_weight=None)
```
本函数在一个batch的样本上对模型进行评估

函数的返回与evaluate的情形相同

###predict_on_batch
```python
predict_on_batch(self, x)
```
本函数在一个batch的样本上对模型进行测试

函数返回模型在一个batch上的预测结果

###fit_generator
```python
fit_generator(self, generator, samples_per_epoch, nb_epoch, verbose=1, callbacks=[], validation_data=None, nb_val_samples=None, class_weight=None, max_q_size=10)
```
利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练

函数的参数是：

generator：生成器函数，生成器的输出应该为：

一个形如（inputs，targets）的tuple
一个形如（inputs, targets,sample_weight）的tuple。所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
samples_per_epoch：整数，当模型处理的样本达到此数目时计一个epoch结束，执行下一个epoch
verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
validation_data：具有以下三种形式之一

生成验证集的生成器
一个形如（inputs,targets）的tuple
一个形如（inputs,targets，sample_weights）的tuple
nb_val_samples：仅当validation_data是生成器时使用，用以限制在每个epoch结束时用来验证模型的验证集样本数，功能类似于samples_per_epoch
max_q_size：生成器队列的最大容量
函数返回一个History对象

例子：

```python
def generate_arrays_from_file(path):
    while 1:
        f = open(path)
        for line in f:
            # create numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_line(line)
            yield (x, y)
        f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        samples_per_epoch=10000, nb_epoch=10)
        
```

###evaluate_generator
```python
evaluate_generator(self, generator, val_samples, max_q_size=10)
```
本函数使用一个生成器作为数据源评估模型，生成器应返回与test_on_batch的输入数据相同类型的数据。该函数的参数与fit_generator同名参数含义相同



#19 损失函数

损失函数（或称目标函数、优化评分函数）是编译模型时所需的两个参数之一：

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```
```python
from keras import losses
model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```
你可以传递一个现有的损失函数名，或者一个 TensorFlow/Theano 符号函数。 该符号函数为每个数据点返回一个标量，有以下两个参数:

y_ true: 真实标签。TensorFlow/Theano 张量。
y_ pred: 预测值。TensorFlow/Theano 张量，其 shape 与 y_true 相同。
实际的优化目标是所有数据点的输出数组的平均值。

##可用损失函数

```python
mean_squared_error

mean_squared_error(y_true, y_pred)
mean_absolute_error

mean_absolute_error(y_true, y_pred)
mean_absolute_percentage_error

mean_absolute_percentage_error(y_true, y_pred)
mean_squared_logarithmic_error

mean_squared_logarithmic_error(y_true, y_pred)
squared_hinge

squared_hinge(y_true, y_pred)
hinge

hinge(y_true, y_pred)
categorical_hinge

categorical_hinge(y_true, y_pred)
logcosh

logcosh(y_true, y_pred)
预测误差的双曲余弦的对数。

对于小的 x，log(cosh(x)) 近似等于 (x ** 2) / 2。对于大的 x，近似于 abs(x) - log(2)。这表示 'logcosh' 与均方误差大致相同，但是不会受到偶尔疯狂的错误预测的强烈影响。

参数

y_true: 目标真实值的张量。
y_pred: 目标预测值的张量。
返回

每个样本都有一个标量损失的张量。

categorical_crossentropy

categorical_crossentropy(y_true, y_pred)
sparse_categorical_crossentropy

sparse_categorical_crossentropy(y_true, y_pred)
binary_crossentropy

binary_crossentropy(y_true, y_pred)
kullback_leibler_divergence

kullback_leibler_divergence(y_true, y_pred)
poisson

poisson(y_true, y_pred)
cosine_proximity

cosine_proximity(y_true, y_pred)
```
注意: 当使用 categorical_ crossentropy 损失时，你的目标值应该是分类格式 (即，如果你有 10 个类，每个样本的目标值应该是一个 10 维的向量，这个向量除了表示类别的那个索引为 1，其他均为 0)。 为了将 整数目标值 转换为 分类目标值，你可以使用 Keras 实用函数 to_ categorical：

```python
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(int_labels, num_classes=None)
```

#20 评价函数
评价函数用于评估当前训练模型的性能。当模型编译后（compile），评价函数应该作为 metrics 的参数来输入。

```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])

```
评价函数和 损失函数 相似，只不过评价函数的结果不会用于训练过程中。

我们可以传递已有的评价函数名称，或者传递一个自定义的 Theano/TensorFlow 函数来使用（查阅自定义评价函数）。

###参数
* y_ true: 真实标签，Theano/Tensorflow 张量。
* y_ pred: 预测值。和 y_ true 相同尺寸的 Theano/TensorFlow 张量。

###返回值
返回一个表示全部数据点平均值的张量。

##可使用的评价函数
...



#21 optimizer
优化器 (optimizer) 是编译 Keras 模型的所需的两个参数之一：

```python
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```
你可以先实例化一个优化器对象，然后将它传入 model.compile()，像上述示例中一样， 或者你可以通过名称来调用优化器。在后一种情况下，将使用优化器的默认参数。

```python
# 传入优化器名称: 默认参数将被采用
model.compile(loss='mean_squared_error', optimizer='sgd')
```

##21.1 Keras 优化器的公共参数

参数 `clipnorm` 和 `clipvalue` 能在所有的优化器中使用，用于控制梯度裁剪（Gradient Clipping）：

```python
from keras import optimizers

# 所有参数梯度将被裁剪，让其l2范数最大为1：g * 1 / max(1, l2_norm)
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```
```python
from keras import optimizers

# 所有参数d 梯度将被裁剪到数值范围内：
# 最大值0.5
# 最小值-0.5
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```

##21.2 SGD
```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```
随机梯度下降优化器。
包含扩展功能的支持： - 动量（momentum）优化, - 学习率衰减（每次参数更新后） - Nestrov 动量 (NAG) 优化

###参数

* lr: float >= 0. 学习率。
* momentum: float >= 0. 参数，用于加速 SGD 在相关方向上前进，并抑制震荡。
* decay: float >= 0. 每次参数更新后学习率衰减值。
* nesterov: boolean. 是否使用 Nesterov 动量。


##21.3 RMSprop
```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
```
RMSProp 优化器.
建议使用优化器的默认参数 （除了学习率 lr，它可以被自由调节）
这个优化器通常是训练循环神经网络RNN的不错选择。

###参数

* lr: float >= 0. 学习率。
* rho: float >= 0. RMSProp梯度平方的移动均值的衰减率.
* epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
* decay: float >= 0. 每次参数更新后学习率衰减值。


##21.4 Adagrad
```python
keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
```
Adagrad 优化器。
Adagrad 是一种具有特定参数学习率的优化器，它根据参数在训练期间的更新频率进行自适应调整。参数接收的更新越多，更新越小。

建议使用优化器的默认参数。

###参数
* lr: float >= 0. 学习率.
* epsilon: float >= 0. 若为 None, 默认为 K.epsilon().
* decay: float >= 0. 每次参数更新后学习率衰减值.

##21.5 Adadelta
```python
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
```
Adadelta 优化器。
Adadelta 是 Adagrad 的一个具有更强鲁棒性的的扩展版本，它不是累积所有过去的梯度，而是根据渐变更新的移动窗口调整学习速率。 这样，即使进行了许多更新，Adadelta 仍在继续学习。 与 Adagrad 相比，在 Adadelta 的原始版本中，您无需设置初始学习率。 在此版本中，与大多数其他 Keras 优化器一样，可以设置初始学习速率和衰减因子。

建议使用优化器的默认参数。

###参数

* lr: float >= 0. 学习率，建议保留默认值。
* rho: float >= 0. Adadelta梯度平方移动均值的衰减率。
* epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
* decay: float >= 0. 每次参数更新后学习率衰减值。


##21.6 Adam
```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
```
Adam 优化器。
默认参数遵循原论文中提供的值。

###参数

* lr: float >= 0. 学习率。
* beta_1: float, 0 < beta < 1. 通常接近于 1。
* beta_2: float, 0 < beta < 1. 通常接近于 1。
* epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
* decay: float >= 0. 每次参数更新后学习率衰减值。
* amsgrad: boolean. 是否应用此算法的 AMSGrad 变种，来自论文 "On the Convergence of Adam and Beyond"。

##21.7 Adamax

##21.8 Nadam


#22 激活函数的用法

激活函数可以通过设置单独的激活层实现，也可以在构造层对象时通过传递 activation 参数实现：

```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```
等价于：

```python
model.add(Dense(64, activation='tanh'))
```
你也可以通过传递一个逐元素运算的 Theano/TensorFlow/CNTK 函数来作为激活函数：

```python
from keras import backend as K

model.add(Dense(64, activation=K.tanh))
model.add(Activation(K.tanh))
```

##预定义激活函数

###softmax
```python
keras.activations.softmax(x, axis=-1)
```
Softmax 激活函数。

参数

x：张量。
axis：整数，代表softmax所作用的维度。
返回

softmax 变换后的张量。

异常


###elu
```python
keras.activations.elu(x, alpha=1.0)
```
指数线性单元。

参数

x：张量。
alpha：一个标量，表示负数部分的斜率。
返回

线性指数激活：如果 x > 0，返回值为 x；如果 x < 0 返回值为 alpha * (exp(x)-1)

###selu
```python
keras.activations.selu(x)
```
可伸缩的指数线性单元（SELU）。

SELU 等同于：scale * elu(x, alpha)，其中 alpha 和 scale 是预定义的常量。只要正确初始化权重（参见 lecun_normal 初始化方法）并且输入的数量「足够大」（参见参考文献获得更多信息），选择合适的 alpha 和 scale 的值，就可以在两个连续层之间保留输入的均值和方差。

参数

x: 一个用来用于计算激活函数的张量或变量。
返回

可伸缩的指数线性激活：scale * elu(x, alpha)。

注意

与「lecun_normal」初始化方法一起使用。
与 dropout 的变种「AlphaDropout」一起使用。
参考文献

Self-Normalizing Neural Networks

###softplus
```python
keras.activations.softplus(x)
```
Softplus 激活函数。

参数

x: 张量。
返回

Softplus 激活：log(exp(x) + 1)。

###softsign
```python
keras.activations.softsign(x)
```
Softsign 激活函数。

参数

x: 张量。
返回

Softsign 激活：x / (abs(x) + 1)。

###relu
```python
keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```
整流线性单元。

使用默认值时，它返回逐元素的 max(x, 0)。

否则，它遵循：

如果 x >= max_value：f(x) = max_value，
如果 threshold <= x < max_value：f(x) = x，
否则：f(x) = alpha * (x - threshold)。
参数

x: 张量。
alpha：负数部分的斜率。默认为 0。
max_value：输出的最大值。
threshold: 浮点数。Thresholded activation 的阈值值。

返回
一个张量。

###tanh
```python
keras.activations.tanh(x)
```
双曲正切激活函数。

###sigmoid

`sigmoid(x)`

Sigmoid 激活函数。

###hard_sigmoid

hard_sigmoid(x)

Hard sigmoid 激活函数。

计算速度比 sigmoid 激活函数更快。

参数

x: 张量。
返回

Hard sigmoid 激活：

如果 x < -2.5，返回 0。
如果 x > 2.5，返回 1。
如果 -2.5 <= x <= 2.5，返回 0.2 * x + 0.5。

###exponential
```python
keras.activations.exponential(x)
```
自然数指数激活函数。

###linear
```python
keras.activations.linear(x)
```
线性激活函数（即不做任何改变）


#23 callback函数
...


#24 常用数据集
##24.1 CIFAR10 小图像分类数据集

50,000 张 32x32 彩色训练图像数据，以及 10,000 张测试图像数据，总共分为 10 个类别。

用法：

```python
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```
返回：
2 个元组：
x_ train, x_ test: uint8 数组表示的 RGB 图像数据，尺寸为 (num_ samples, 3, 32, 32) 或 (num_ samples, 32, 32, 3)，基于 image_ data_ format 后端设定的 channels_ first 或 channels_ last。
y_ train, y_ test: uint8 数组表示的类别标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)。


##24.2 CIFAR100 小图像分类数据集

50,000 张 32x32 彩色训练图像数据，以及 10,000 张测试图像数据，总共分为 100 个类别。

用法：

```python
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
```
返回：

2 个元组：
x_train, x_test: uint8 数组表示的 RGB 图像数据，尺寸为 (num_samples, 3, 32, 32) 或 (num_samples, 32, 32, 3)，基于 image_data_format 后端设定的 channels_first 或 channels_last。
y_train, y_test: uint8 数组表示的类别标签，尺寸为 (num_samples,)。

参数：
label_mode: "fine" 或者 "coarse"

##24.4 IMDB 电影评论情感分类数据集

数据集来自 IMDB 的 25,000 条电影评论，以情绪（正面/负面）标记。评论已经过预处理，并编码为词索引（整数）的序列表示。为了方便起见，将词按数据集中出现的频率进行索引，例如整数 3 编码数据中第三个最频繁的词。这允许快速筛选操作，例如：「只考虑前 10,000 个最常用的词，但排除前 20 个最常见的词」。

作为惯例，0 不代表特定的单词，而是被用于编码任何未知单词。

用法

```python
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

```
返回：

2 个元组：
x_train, x_test: 序列的列表，即词索引的列表。如果指定了 num_words 参数，则可能的最大索引值是 num_words-1。如果指定了 maxlen 参数，则可能的最大序列长度为 maxlen。
y_train, y_test: 整数标签列表 (1 或 0)。

参数:

path: 如果你本地没有该数据集 (在 '~/.keras/datasets/' + path)，它将被下载到此目录。
num_words: 整数或 None。要考虑的最常用的词语。任何不太频繁的词将在序列数据中显示为 oov_char 值。
skip_top: 整数。要忽略的最常见的单词（它们将在序列数据中显示为 oov_char 值）。
maxlen: 整数。最大序列长度。 任何更长的序列都将被截断。
seed: 整数。用于可重现数据混洗的种子。
start_char: 整数。序列的开始将用这个字符标记。设置为 1，因为 0 通常作为填充字符。
oov_char: 整数。由于 num_words 或 skip_top 限制而被删除的单词将被替换为此字符。
index_from: 整数。使用此数以上更高的索引值实际词汇索引的开始。


##24.5 路透社新闻主题分类

数据集来源于路透社的 11,228 条新闻文本，总共分为 46 个主题。与 IMDB 数据集一样，每条新闻都被编码为一个词索引的序列（相同的约定）。

用法：

```python
from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
                                                         
```
规格与 IMDB 数据集的规格相同，但增加了：

test_split: 浮点型。用作测试集的数据比例。
该数据集还提供了用于编码序列的词索引：

`word_index = reuters.get_word_index(path="reuters_word_index.json")`

返回： 一个字典，其中键是单词（字符串），值是索引（整数）。 例如，word_index["giraffe"] 可能会返回 1234。

参数：

path: 如果在本地没有索引文件 (at '~/.keras/datasets/' + path), 它将被下载到该目录。



##24.6 MNIST 手写字符数据集

训练集为 60,000 张 28x28 像素灰度图像，测试集为 10,000 同规格图像，总共 10 类数字标签。

用法：

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
返回：

2 个元组：
x_train, x_test: uint8 数组表示的灰度图像，尺寸为 (num_samples, 28, 28)。
y_train, y_test: uint8 数组表示的数字标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)。

参数：

path: 如果在本地没有索引文件 (at '~/.keras/datasets/' + path), 它将被下载到该目录。


##24.7 Fashion-MNIST 时尚物品数据集

训练集为 60,000 张 28x28 像素灰度图像，测试集为 10,000 同规格图像，总共 10 类时尚物品标签。该数据集可以用作 MNIST 的直接替代品。类别标签是：

|类别|描述|中文|
|---|---|----|
|0 |T-shirt/top |T恤/上衣 |
|1	|Trouser	|裤子|
|2	|Pullover	|套头衫
|3	|Dress	|连衣裙
|4	|Coat	|外套
|5	|Sandal	|凉鞋
|6	|Shirt	|衬衫
|7	|Sneaker	|运动鞋
|8	|Bag	|背包
|9	|Ankle boot	|短靴


用法：

```python
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```
返回：
2 个元组：
x_train, x_test: uint8 数组表示的灰度图像，尺寸为 (num_samples, 28, 28)。
y_train, y_test: uint8 数组表示的数字标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)。


##24.8 Boston 房价回归数据集

数据集来自卡内基梅隆大学维护的 StatLib 库。

样本包含 1970 年代的在波士顿郊区不同位置的房屋信息，总共有 13 种房屋属性。 目标值是一个位置的房屋的中值（单位：k$）。

用法：

```python
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```
参数：

path: 缓存本地数据集的位置 (相对路径 ~/.keras/datasets)。
seed: 在计算测试分割之前对数据进行混洗的随机种子。
test_split: 需要保留作为测试数据的比例。

返回： Numpy 数组的元组: (x_train, y_train), (x_test, y_test)。


#25 Applications
...

#26 初始化 Initializers
...

#27 正则化 Regularizers
正则化器允许在优化过程中对层的参数或层的激活情况进行惩罚。 网络优化的损失函数也包括这些惩罚项。

惩罚是以层为对象进行的。具体的 API 因层而异，但 Dense，Conv1D，Conv2D 和 Conv3D 这些层具有统一的 API。

正则化器开放 3 个关键字参数：

* `kernel_regularizer`: keras.regularizers.Regularizer 的实例
* `bias_regularizer`: keras.regularizers.Regularizer 的实例
* `activity_regularizer`: keras.regularizers.Regularizer 的实例


例

```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
                
```
可用的正则化器

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```

开发新的正则化器

任何输入一个权重矩阵、返回一个损失贡献张量的函数，都可以用作正则化器，例如：

```python
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg))

```
另外，你也可以用面向对象的方式来编写正则化器的代码


#28 可视化 Visualization
`keras.utils.vis_utils` 模块提供了一些绘制 Keras 模型的实用功能(使用 graphviz)。

以下实例，将绘制一张模型图，并保存为文件：

```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

plot_model 有 4 个可选参数:

* show_shapes (默认为 False) 控制是否在图中输出各层的尺寸。
* show_layer_names (默认为 True) 控制是否在图中显示每一层的名字。
* expand_dim（默认为 False）控制是否将嵌套模型扩展为图形中的聚类。
* dpi（默认为 96）控制图像 dpi。
* 
此外，你也可以直接取得 pydot.Graph 对象并自己渲染它。 例如，ipython notebook 中的可视化实例如下：

```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

```

训练历史可视化
Keras Model 上的 `fit() `方法返回一个` History `对象。`History.history `属性是一个记录了连续迭代的训练/验证（如果存在）损失值和评估值的字典。这里是一个简单的使用 matplotlib 来生成训练/验证集的损失和准确率图表的例子：

```python
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
