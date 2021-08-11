# Week1  Introduction to Deep learning

# Introduction to Deep learning

## 1. what is a Neural Netwiork

Neural Network :  인간의 뇌를 모방하여 스스로 모델을 만드는 인공지능

 

- **example**

coursera 강의에 나온 예시 - Housing Price Prediction

![Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12.06.38.png](Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12.06.38.png)

다음은 6개의 데이터(size of house, price)를 가지고 미래의 집값을 예측하는 모델이다.  이때 size of house를 입력으로 오른쪽 그림의 파란 실선과 같은 일차 함수를 만드는 것이 neural network에서의 neuron의 역할이다 

![Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12.06.46.png](Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12.06.46.png)

이때 아래의 그림처럼 input으로 size 하나의 값이 아닌 여러 특징의 값을 입력으로 사용할 수 있다.  input Layer(x1,x2,x3,x4)와 Ouput Layer 를 제외한 나머지 Layer를 hidden layer라고 한다.

- Neural network에 input 과 output을 줘서 Hidden Unit을 만들고 학습시킴

![Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12.25.00.png](Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12.25.00.png)

## 2. Supervised Learning with Neural Network

Supervised Learning                                                                         - 지도학습은 input data와 정답(lable)을 알려주고 학습을 시키는 것으로 분류(classification)나 회귀(regression)문제가 이에 속한다.          

- 분류 (classification)문제: 예측하는 값의 category(class)를 판별하는 것 이다.

       즉, 결과가 어느 분류에 속하는지를 판단하는 것.

- 회귀(regression) : 위의 집값 예시처럼 예측하는 값(출력)이 continuos하다.

 

![Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.43.41.png](Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1.43.41.png)

위의 표와 같이 지도 학습을 위해 다양한 유형의 Neural Network를 사용하게 된다.

- 소리를 text로 바꿔주는 경우나 영어를 중국어로 번역하는 경우와 같이 시간적 요소가 존재하는 1차원 시퀀스 데이터에서는 RNN(Recurrent Neural Nerwork)를 주로 사용한다.
- image processing에서는 CNN(Conventional Neural Network)를 주로 사용한다.
- 자율주행 자동차와 같이 이미지, 이외의 여러 센서 정보들이 입력으로 들어가는 경우에는 Custom Version 이나 Hybrid Neural Network를 사용하게 된다

 

**Structured Data VS Unstructured Data**

Structured Data (정형 데이터) :  구조와 관리 체계의 규칙이 정해져있는 데이터. 즉, data의 database          

Unstructured Data (비정형 데이터) : 음성이나 텍스트, 영상처럼 정해진 규칙이 없는 데이터

- Unstructured Data의 경우 computer가 인식하기 어렵다

      → 최근 딥러닝의 발달로 인해 점차 인식률이 좋아지고 있음

- NN의 경제적인 value의 경우 Structured Data에 기반한다.

## 3. Why is Deep Learning taking off?

- Iot의 발달, 모바일 폰 사용량 증가 등 사회가 디지털화 되면서 Data의 양이 점차 방대해짐
- GPU 및 HW 발전에 따른 Computation 능력 향상

      → 기존의 학습이 오래걸리던 Neural Network를 빠른시간에 처리하게 됌

- Algorithms의 혁신

   ex) sigmoid → ReLu 

     

![Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.17.39.png](Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.17.39.png)

  sigmoid 

![Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.17.29.png](Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.17.29.png)

ReLu(rectified linear units)

기존에 사용하던 sigmoid의 경우 양끝쪽으로 갈 수록 기울기가 0에 가까워짐 → gradient descent 도입 시 기울기 값이 너무 작아 학습이 굉장히 느리게 진행됌 

(1개 이상의 hidden layer를 넣기 힘듬)

→ ReLu 사용시 여러 hidden layer를 넣었을 경우에도 모델 학습이 진행됌 

**Scale Drives Deep learning Progress**

![Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.08.43.png](Week1%20Introduction%20to%20Deep%20learning%20264224a604ab404697ce20697bbcb5b6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.08.43.png)

labed data = 입력과 출력값 쌍

- Training Data Set이 작은 경우 학습 알고리즘의 성능 차이가 거의 없음 → 알고리즘 내부의 parameter 조정이 성능에 영향을 줌
- Training Data Set의 크기가 클수록 전통적인 알고리즘과 Neural Network 와의 학습 성능 차이가 커짐
- Training Data Set의 크기가 클수록  큰 신경망의 학습 성능이 좋다

### 끝으로

현재 성능이 좋은 모델들이 많이 있지만, 새로운 문제를 직면하게 된다면 기존의 모델을 바로 적용시키기 보다는 모델의 특성을 이용해서 다양한 아이디어를 먼저 생각해보면 좋을 것 같다.