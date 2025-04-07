#머신러닝
#머신러닝은 데이터를 통해 알고리즘이 스스로 학습하도록 하는 것
#데이터 패턴을 찾고 예측, 결정 내림

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


import numpy as np
from scipy.stats import alpha
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(1)
#남성과 여성의 키, 체중 데이터 생성
h_w = np.random.normal(160,5,50) #키 평균 160 표준편차 5, 50개
w_w = np.random.normal(50,5,50)

h_m = np.random.normal(175,7,50)
w_m = np.random.normal(70,7,50)

X = np.vstack((np.column_stack((h_w,w_w)),np.column_stack((h_m,w_m))))
y = np.array([0]*50+[1]*50) #여성을 0 남성을 1 로 레이블링 (정답지)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

plt.scatter(X_train[:,0], X_train[:,1],c=y_train,marker='o',label='train')
plt.scatter(X_test[:,0], X_test[:,1],c=y_test,marker='x',label='test')

x_min, x_max = X[:,0].min() -1, X[:,0].max() +1
y_min, y_max = X[:,1].min() -1, X[:,1].max() +1

xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
Z = knn.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx,yy,Z,alpha=0.3)
plt.legend()
plt.xlabel('h')
plt.ylabel('x')

plt.show()
