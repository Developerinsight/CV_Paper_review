# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Link: https://arxiv.org/pdf/1506.01497.pdf
## Module
### 1. RPN(Region Proposal Network)


<img width="610" alt="image" src="https://github.com/Developerinsight/Paper_review/assets/123748877/915b7223-2d53-41be-a9a9-0da0551c16f3">

#### 1.1 합성곱 계층을 활용해 전체 이미지에 합성곱 연산
#### 1.2 Feature map을 RPN, Classifier에 전달(피쳐 맵 공유)
#### 1.3 RPN은 Feature Map 기반으로 객체가 있을 만한 곳 찾아준다.(영역 추정)
##### 1.3.1 How to RPN
<img width="482" alt="image" src="https://github.com/Developerinsight/Paper_review/assets/123748877/21415373-4da0-42f2-b5e4-52ff6521aa84"> 

###### W*H 마다의 각 위치마다 K개의 AnchorBox 생성 
###### Regression layer는 (center_x, center_y, w, h) 4k개 + Classification layer는 (Positive Object, Negative Object) 2k개

##### 1.3.2 Loss Function
<img width="569" alt="image" src="https://github.com/Developerinsight/Paper_review/assets/123748877/31df0ae3-60f3-4ddc-81a7-563964d8dd60">

###### 분류손실: 모델이 각 객체의 클래스를 얼마나 잘 예측하는지
###### 회귀손실: 모델이 객체의 위치를 얼마나 잘 예측하는
###### SGD(확률적 경사 하강법) + 역전파 => end to end Training
###### 대부분 이미지에 배경이 많기에 손실 함수 계산 시 Postive Anchor:Negative Anchor = 1:1

### 2. Fast R-CNN
