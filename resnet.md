f(x) = h(x) - x, input인 x와 output인 h(x)의 차이를 학습한다. 
identity mapping(기본 매핑)이 optimal하다면 다중 비선형 레이어 가중치는 zero가 될 것이다.
층이 깊어질수록 잔차함수 f(x)가 학습해야 할 변화가 작고, 따라서 이는 H(x)인 항등 함수(항등 매핑)에 가까운 함수를 학습하고 있다는 걸 의미한다.
H(x)가 x와 유사하다는 가정

foward propagation에서 f(x) + x (즉, h(x))를 학습하고, back propagation에서 f(x)를 

import torch
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        두 개의 선형(완전연결) 층
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x):
        # 입력 x를 첫 번째 선형 층에 통과시킴
        residual = self.linear1(x)
        # 비선형 활성화 함수 적용 (예: ReLU)
        residual = nn.ReLU()(residual)
        # 두 번째 선형 층 통과
        residual = self.linear2(residual)
        # 스킵 연결: 입력 x를 잔차와 더함
        out = residual + x
        return out

잔차 블록 인스턴스 생성
res_block = ResidualBlock()

# 역전파 과정을 포함하여 가중치를 조정하는 과정을 구현해 보겠습니다.
# 먼저, 손실 함수와 옵티마이저를 정의하겠습니다.

# 손실 함수 (예: 평균 제곱 오차)
loss_function = nn.MSELoss()

# 옵티마이저 (예: SGD)
optimizer = torch.optim.SGD(res_block.parameters(), lr=0.01)

# 타겟 데이터 생성 (임의의 값)
target = torch.rand(10)

# 순전파: 입력 x를 통해 출력 H(x) 생성
output = res_block(x)

# 손실 계산: 출력 H(x)와 타겟 간의 차이
loss = loss_function(output, target)

# 역전파 전에 옵티마이저의 기존 그래디언트를 초기화
optimizer.zero_grad()

# 역전파: 손실에 대한 모델의 모든 가중치에 대한 그래디언트 계산
loss.backward()
=> 역전파 수행하며 손실함수에 대한 네트워크 가중치 그래디언트 계산.
=> 출력 H(x)에서 입력 x로 거슬러 올라가며 가중치 손실 기울기 계산. 이 때 F(x)에 대한 가중치도 포함. 

# 옵티마이저를 사용하여 가중치 업데이트
optimizer.step()

print("Loss:", loss.item())


# 역전파 통해 fx를 0에 가깝게 만드는 것??

