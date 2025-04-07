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


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression #선형회귀 클래스 사용
from sklearn.model_selection import train_test_split #훈련셋 테스트셋
from sklearn.datasets import make_regression #회귀 데이터셋 만드는 함수

# 1.데이터의 생성
#make_regression 함수는 회귀 예측 가상 데이터 생성

X,y = make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
#n_samples : 샘플의 수 ( 데이터의 수 )
#n_features : 각 데이터 특성 ( 독립변수의 수 )
#noise : 잡음을 넣어서 데이터 패턴의 불규칙성을 줌, 노이즈가 너무 크다면 학습 어려움
#random_state : 난수 생성 기준 값

X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
#트레인 테스트 스플릿이 학습용과 테스트용 데이터로 나눠줌
#0.2비율 만큼 테스트데이터를 분할 나머지 80%는 학습용으로 사용 예정
model = LinearRegression() #선형회귀 모델 객체 선언
#위에서 만든 데이터셋을 이 model에다가 넣어서 훈련 시킬 예정
#모델학습은 fit으로 학습시킨다.
#fit()
model.fit(X_train,y_train)
#예측
y_pred = model.predict(X_test)
plt.scatter(X_test, y_test, color='blue', label='value')
plt.plot(X_test, y_pred, color = 'red',linewidth=2, label='pred')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
print(model.score(X_test,y_test))
#결정계수 : 모델이 얼마나 잘 추측하는가 0~1 사이 값 ( 정확도)
