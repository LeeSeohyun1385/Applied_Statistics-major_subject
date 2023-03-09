"""
       Week 10
   
"""
#클래스 불균형 
#tp/fp는 크고, tn/fn은 작았으면 좋겠음
#recall: 불균형 있을시 왜곡 -> 중요한 측도 
#f1-score 가 accuracy보다 더 많이 사용

# Fashion MNIST : 10 종류의 의류상품 분류
# 참고 : https://www.tensorflow.org/tutorials/keras/classification?hl=ko
#"./fashion-mnist-sprite.png"

from IPython.display import Image
Image("C:/Users/Lee seohyun/Downloads/fashion-mnist-sprite/fashion-mnist-sprite.png", width = 640)#, height = 300)

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


# data import

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()   


# class name

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#데이터 탐색 
print(train_images.shape)#60000개, 28*28픽셀
print(train_labels.shape)

print(test_images.shape)#10000개
print(test_labels.shape)


# example data
#훈련세트의 첫번째 이미지
plt.figure()
plt.imshow(train_images[0]) #아무거나 #imshow로 그려서 파란색 , 원래는 회색조 데이터(0이 검정)
plt.colorbar() #픽셀값의 범위 0~250
plt.grid(False)
plt.show()


# 전처리 : min-max normalization
# 픽셀값 범위 0~250 -> 0~1
train_images = train_images / 255.0

test_images = test_images / 255.0
#0~1

# example data #정규화된 데이터 25개 출력해봄
plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1) #여러개의 그래프 (row, column, index)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary) #회색조
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Tunning parameters

train_examples = train_labels.shape[0] #60000개
batch_size = 10 

epochs = 10 

learning_rate = 1E-3


# tf dataset 생성
#data pipeline구성

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(train_examples).batch(batch_size).repeat()


# 모형 생성
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)), #784픽셀의 1차원 배열로 변환
    tf.keras.layers.Dense(256, activation = 'relu'), #이 hidden layer없으면 logistic회귀
    tf.keras.layers.Dense(10) #10개의 분류확률 반환 (목표변수10개)
])
#반환된 확률 전체합 = 1
#각 노드는 현재 이미지가 10개 클래스 중 하나에 속할 확률을 출력

#인코딩 : 784(28*28)개 데이터를 256개있는것 , 256으로 줄임


#모델 컴파일 
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True) 

optimizer = tf.keras.optimizers.Adam(learning_rate)

model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

model.summary()
#accuracy : 올바르게 분류된 이미지의 비율

# 학습
learning_history = model.fit(train_ds, #slice 안하면 train_images, train_labels
                             epochs = epochs, 
                             steps_per_epoch = train_examples / batch_size, 
                             verbose = 1)

'''정확도 평가'''
#test 정확도 
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
#train 정확도 > test 정확도 -> 차이는 과대적합(overfitting) 

'''예측 : inference'''
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()
                                         ])
#로짓을 확률로 계산해줌 : softmax layer 

predictions = probability_model.predict(test_images)
#test set에 있는 각 이미지의 레이블 예측

print(predictions[0]) #첫번째 예측값 : 10개 품목에 해당하는 모델의 신뢰도 
print(np.argmax(predictions[0])) #가장 높은 신뢰도를 가진 레이블 
#첫번째 이미지 -> 레이블9에 해당 

print(predictions[1]) #두번째 범주 
print(np.argmax(predictions[1]))

print(test_labels[0])#예측이 맞는지 test label 확인 
print(test_labels[1])

tf.math.confusion_matrix(test_labels, np.argmax(predictions, axis=-1))
#6번째(shirts)는 잘 못맞춤(220) -> shirts 와 t-shirts를 헷갈림 
#117 2번째(열)인데 6번째(행)라고 예측 -> 잘 못맞춤 

#baseline model , banila model 







