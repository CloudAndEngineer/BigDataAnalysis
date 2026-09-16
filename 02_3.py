# [프로그램 2-1] 선형 분류 모델 구현하기
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 데이터 생성
# shape: X (100, 2), y (100,)

torch.manual_seed(0)
# 파이토치(PyTorch)에서 난수(랜덤 숫자) 생성기의 시드(Seed)를 고정하여 실험 결과를 재현할 수 있게 만드는 함수
# Seed: 난수 생성기의 초기값으로, 동일한 Seed를 사용하면 동일한 난수 시퀀스를 생성할 수 있음
# 시드를 고정하지 않으면 실행할 때마다 성능이 들쭉날쭉 변함
# 내가 바꾼 코드 때문에 성능이 좋아진 것인지, 아니면 그냥 운 좋게 좋은 랜덤 숫자가 뽑힌 것인지 구별하기 위해 시드를 고정함

X = torch.randn(100, 2)
# 100행 2열의 난수 행렬 생성

y = (X[:, 0] + X[:, 1] > 0).long()
# x[:, 0]의 의미: X의 모든 행에서 첫 번째 열(0번째 인덱스)을 선택
# ... > 0: x1 + x2 > 0이면 1, 아니면 0
# long(): 정수형으로 변환

# 예시
# X = tensor([
#    [2, -3],
#    [-1, 4],
#    [-5, 2],
#    [1, -3]
# ])
# x[:, 0] = tensor([2, -1, -5, 1])
# x[:, 1] = tensor([-3, 4, 2, -3])
# x[:, 0] + x[:, 1] = tensor([-1, 3, -3, -2])
# y = (X[:, 0] + X[:, 1] > 0).long() = tensor([0, 1, 0, 0])

# 모델 정의
model = nn.Sequential(
    nn.Linear(2, 1),      # shape: (batch, 1)
    nn.Sigmoid()
)
# Sequential: 여러 계층을 순차적으로 쌓아 모델을 구성할 수 있는 클래스
# 모델은 nn.Sequential을 사용하여 순차적으로 구성됨

# nn.Linear(2, 1): 입력 차원이 2이고 출력 차원이 1인 선형 계층
# 입력 데이터의 차원의 개수는 상관없다. 마지막 차원의 크기만 맞으면 됨. (batch, 2) -> (batch, 1)
# nn.Sigmoid(): 시그모이드 활성화 함수 적용

criterion = nn.BCELoss()
# 이진 분류(Binary Classification) 문제에서 사용되는 손실 함수(Loss Function)로, 모델의 출력과 실제 레이블 간의 차이를 측정하는 데 사용됨
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 최적화 알고리즘 사용

loss_history = []

# 학습
for epoch in range(50):
    optimizer.zero_grad()
    # Initialize gradients to zero
    # If you don't do this, gradients will accumulate from previous iterations, leading to incorrect updates.

    output = model(X).squeeze() # shape: (100,)
    # Role of model(X): The model takes the input data X and produces an output.
    # In this case, the model is a simple neural network with one linear layer followed by a sigmoid activation function.
    # The output will be a tensor of shape (100, 1) because the linear layer has one output unit.

    # squeeze(): reduce dimension, (100, 1) -> (100,)
    # In this case, matrix(100, 1) is converted to vector(100,) by removing the second dimension.

    loss = criterion(output, y.float())
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())


# 손실 곡선 시각화
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 분류 결과 시각화
with torch.no_grad():
    preds = (model(X).squeeze() > 0.5).long()

plt.scatter(X[:, 0], X[:, 1], c=preds, cmap='coolwarm')
plt.title("Classification Result")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# [프로그램 2-2] K-means 군집화와 시각화하기
import torch
import matplotlib.pyplot as plt

# 데이터 생성
# shape: (200, 2)
torch.manual_seed(0)
X = torch.cat([
    torch.randn(100, 2) + torch.tensor([2.0, 0.0]),
    torch.randn(100, 2) + torch.tensor([-2.0, 0.0])
], dim=0)

# 초기 중심점 선택
# shape: (2, 2)
centroids = X[torch.randperm(X.size(0))[:2]]

for _ in range(20):
    # 각 점이 가장 가까운 중심점을 선택
    distances = torch.cdist(X, centroids)    # shape: (200, 2)
    labels = distances.argmin(dim=1)         # shape: (200,)
    
    # 새 중심점 계산
    new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(2)])
    
    if torch.allclose(centroids, new_centroids):
        break
    centroids = new_centroids

# 군집 결과 시각화
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X')
plt.title("K-means Clustering Result")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
