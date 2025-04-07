#머신러닝
#머신러닝은 데이터를 통해 알고리즘이 스스로 학습하도록 하는 것
#데이터 패턴을 찾고 예측, 결정 내림
from pickletools import optimize

#기법
#감독학습, 수퍼바이스드 러닝
#비감독학습 언수퍼바이스드 러닝
#강화학급 레인포스 러닝

#머신러닝은 상대적으로 작은 규목 데이터나 간단한 모델(선형회귀, 의사결정트리 , KNN 등)

#딥러닝
#딥러닝은 인공신경망 기반으로 하는 것
#뉴럴 네트워크를 사용하여 데이터 분석/예측

#다층 신경만을 이용해서 데이터의 특성(f) 추출하고 대규모 데이터 셋과 복잡한 모델 다룸
#음성처리/nlp자연어 처리/이미지 처리

#AI

#파이썬에서의 딥러닝
#텐서플로우 구글개발 딥러닝
#케라스 텐서플로우에서 제공하는 api
#파이토치 페이스북에서 개발 딥러닝 lib


#파이썬에서의 머신러닝
#사이킷런 : 분류 회귀 군집 등
#XGBoost
#LightGBM.
















































#선형회귀
#데이터의 독립변수와 종속변수 사이의 선형관계를 모델링하는 기법
#예측값을 얻는 목적

#사이킷런을 사용하여 머신러닝 활용
#사이킷런은 분류 회귀 군집 다양한 알고리즙이 있다.
#주요 클래스 fit(학습) predict(예측) score(성능 평가)로 구분

#YOLO


# 선형회귀 예제
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression #선형회귀 클래스 사용
# from sklearn.model_selection import train_test_split #훈련셋 테스트셋
# from sklearn.datasets import make_regression #회귀 데이터셋 만드는 함수
#
# # 1.데이터의 생성
# #make_regression 함수는 회귀 예측 가상 데이터 생성
#
# X,y = make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
# #n_samples : 샘플의 수 ( 데이터의 수 )
# #n_features : 각 데이터 특성 ( 독립변수의 수 )
# #noise : 잡음을 넣어서 데이터 패턴의 불규칙성을 줌, 노이즈가 너무 크다면 학습 어려움
# #random_state : 난수 생성 기준 값
#
# X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
# #트레인 테스트 스플릿이 학습용과 테스트용 데이터로 나눠줌
# #0.2비율 만큼 테스트데이터를 분할 나머지 80%는 학습용으로 사용 예정
# model = LinearRegression() #선형회귀 모델 객체 선언
# #위에서 만든 데이터셋을 이 model에다가 넣어서 훈련 시킬 예정
# #모델학습은 fit으로 학습시킨다.
# #fit()
# model.fit(X_train,y_train)
# #예측
# y_pred = model.predict(X_test)
# plt.scatter(X_test, y_test, color='blue', label='value')
# plt.plot(X_test, y_pred, color = 'red',linewidth=2, label='pred')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.show()
# print(model.score(X_test,y_test))
# #결정계수 : 모델이 얼마나 잘 추측하는가 0~1 사이 값 ( 정확도)





#KNN K nearest neighbor
#최근접 이웃 알고리즘
#머신러닝 사이킷런

# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
#
# X=np.array([[2,80],[3,90],[5,85],[1,60],[4,95],[6,85],[7,90],[2,75],[3,80],[8,90]])
# y=np.array([0,1,1,0,2,2,2,0,1,2])
# # 0 1 2 순으로 높음
#
# X_train,  X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
# model=KNeighborsClassifier(n_neighbors=3)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(classification_report(y_test,y_pred))


# import numpy as np
# from scipy.stats import alpha
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
#
# np.random.seed(1)
# #남성과 여성의 키, 체중 데이터 생성
# h_w = np.random.normal(160,5,50) #키 평균 160 표준편차 5, 50개
# w_w = np.random.normal(50,5,50)
#
# h_m = np.random.normal(175,7,50)
# w_m = np.random.normal(70,7,50)
#
# X = np.vstack((np.column_stack((h_w,w_w)),np.column_stack((h_m,w_m))))
# y = np.array([0]*50+[1]*50) #여성을 0 남성을 1 로 레이블링 (정답지)
#
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train,y_train)
# y_pred = knn.predict(X_test)
# accuracy = accuracy_score(y_test,y_pred)
# print(accuracy)
#
# plt.scatter(X_train[:,0], X_train[:,1],c=y_train,marker='o',label='train')
# plt.scatter(X_test[:,0], X_test[:,1],c=y_test,marker='x',label='test')
#
# x_min, x_max = X[:,0].min() -1, X[:,0].max() +1
# y_min, y_max = X[:,1].min() -1, X[:,1].max() +1
#
# xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
# Z = knn.predict(np.c_[xx.ravel(),yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# plt.contourf(xx,yy,Z,alpha=0.3)
# plt.legend()
# plt.xlabel('h')
# plt.ylabel('x')
#
# plt.show()

# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
#
# cat_size = np.random.randint(30,51,333)
# cat_weight = np.random.randint(3,9,333)
# cat_legs = np.ones(333) * 4
# cat_ear_shape = np.zeros(333)
# cat_food = np.zeros(333)
# X_cat = np.column_stack((cat_size, cat_weight, cat_legs, cat_ear_shape, cat_food))
#
# elephant_size = np.random.randint(200,301,333)
# elephant_weight = np.random.randint(4000,6001,333)
# elephant_legs = np.ones(333) * 4
# elephant_ear_shape = np.ones(333)
# elephant_food = np.ones(333)
# X_elephant = np.column_stack((elephant_size, elephant_weight, elephant_legs, elephant_ear_shape, elephant_food))
#
# dog_size = np.random.randint(100,310,334)
# dog_weight = np.random.randint(10,70,334)
# dog_legs = np.ones(334) * 4
# dog_ear_shape = np.zeros(334)
# dog_food = np.zeros(334)
# X_dog = np.column_stack((dog_size, dog_weight, dog_legs, dog_ear_shape, dog_food))
#
# X = np.vstack((X_cat,X_elephant,X_dog))
# y = np.array([0] * 333 + [1] * 333 + [2] * 334)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)
#
# scaler = StandardScaler()
# #특성 스케일링 (표준화) 데이터 평균을 0 표준편차 1로 맞춘다
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)
# model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, random_state=1)
# #다층퍼센트론 10개뉴런의 2층구조 (은닉층) 히든레이더 max_iter 최대 반복수
# model.fit(X_train_scaled, y_train)
# y_pred = model.predict(X_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"모델 정확도: {accuracy:.5f}")
#
# size = float(input('크기 얼마?'))
# weight = float(input('무게 얼마?'))
# legs = int(input('다리 몇개?'))
# ear = int(input('귀 모양 어떰? 0: 작음 1 : 큼'))
# food = int(input('밥 뭐 먹ㅇ므? 0: 사료 1 : 풀'))
#
# user_data = np.array([[size,weight,legs,ear,food]])
# user_data_scaled = scaler.transform(user_data)
# prediction = model.predict(user_data_scaled)
#
# if prediction == 0:
#     print("고양이인듯")
# elif prediction == 1:
#     print("코끼리인듯")
# else:
#     print("강아지인듯")














#가중치 w
# y=wx+b
# y=wx제곱 + b

#ai 모델 신경망은 학습 과정을 통해 가중치를 조정한다.
#조정하는 과정은 경사하강법을 통해 이루어짐

#초기화 : 모델 처음 만들때 가중치는 랜덤으로 부여 ex# 0.3
#예측 : 모델에 입력 데이터가 들어가면 입력값들이 가중치와 곱해지고 출력이 나옴
#오차 계산 : 모델이 계산한 예측값과 실제(정답)과 차이 계산, 정답이 100 예측 20 => 80
#가중치 업데이트 : 경사하강법을 사용하여 오차가 최소화되는 방향으로 가중치 조절 ( 기울기가 낮아지는 방향)
#가중치 업데이틑 반복



#경사하강법
#경사하강법은 가중치 조정하여 손실함수(loss fuction)를 최소화하는 방법
#경사하강법의 작동 원리
#오차를 최소화 하는 목적이기 때문에 실제값과 예측값 차이를 통해 정의 됨
#mse (mean squared error) 를 손실함수로 사용


#기울기의 계산: 경사하강법은 기울기를 계산하고, 기울기랑 손실함수의 미분값을 의미
#기울기가 크면 가중치를 많이 조정
#기울기가 작으면 조정 폭 조금

#파라미터 업데이트 : 기울기를 계산하고 가중치를 업데이트 함 -> 이 과정을 반복해서 손실함수가 최소화되도록 함

#학습률: 가중치를 얼마나 조정할지를 결정하는 학습률 파라미터, 너무 크면 학습이 불안정, 낮으면 속도 느림

#1. 가중치 : 학습과정이 반복되면서 가중치가 계속 조정,
#2. 학습률 : learning rate 얼마나 큰 폭으로 가중치를 업데이트 할 지
#3. MSE 평균제곱오차 : MSE가 작다 ? => 가중치가 그럴싸함
#4. 기울기는 손실함수가 가중치에 대해 얼마나 민감한지 (큰지)



#모델의 훈련에서 에폭과 배치
#모델 훈련은 데이터를 통해 패턴 학습하는 과정, 데이터셋과 레이블(정답지)를 가지고 예측하고
#그 예측이 얼마나 정확한지 손실함수 계산 후 가중치 업데이트 함

#에포크 epoch
#에폭은 모델이 학습 데이터를 한 번 모두 학습하는 단위 의미
#좋은 학습하려면 여러번 반복 에폭 필요함
#너무 많으면 성능 저하

#배치
#훈련 데이터셋이 크면 모든 데이터를 한번에 담기 어려움
#데이터를 작은 덩어리인 배치로 나누어 처리

#미니배치
#데이터를 여러개 작은 배치로 나누어 각 배치마다 가중치 업데이트 과정을 함

#배치사이즈
#한번에 학습할 데이터 샘플 수 설정 값

#모델 평가 예측
#테스트 데이터셋 (test data 0.2)
#모델 훈련동안 훈련(Train) 데이터에 대해서만 학습함.
#훈련데이터 대한 정확도 높게 훈련이되도 테스트데이터에서 정확도가 낮으면 과적합의심 해봐야함
#테스트데이터는 모델 훈련시 사용하지 않고, 모델이 얼마나 일반화 되었는지


#성능 & 평가 지표 matrics
#정확도 : 모델이 정확히 예측한 샘플의 비율
#정확도 = 예측이 맞은수/ 전체수
#손실함수 : 모델이 예측한 값과 실제 정답과 차이 측정하는 함수

#오버피팅 문제 해결 : 1.데이터 전처리 2.하이퍼파라미터 튜닝 3.데이터 샘플 증가
#드롭아웃 dropout : 네트워크(신경망) 일부 뉴런을 임의로 제한

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# mnist: 숫자에 대한 손글씨 데이터 셋
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 정규화 (0~1 범위로)
X_train, X_test = X_train / 255.0, X_test / 255.0

# 모델 구성
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=5)

# 테스트 정확도
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"{test_acc:.4f} 정확도")

# 예측 결과 출력 + 시각화
y_pred = model.predict(X_test)
print("첫 번째 이미지 예측 결과:", y_pred[0].argmax())
plt.imshow(X_test[0], cmap=plt.cm.binary)
plt.show()
