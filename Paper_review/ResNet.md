이 글은 논문 **Deep Residual Learning for Image Recognition (2015)** 에 대한 설명입니다. 논문 원본에 대한 링크는 아래에 적어놓았습니다.

논문 원본 : https://arxiv.org/abs/1512.03385

# Abstract
* 본 논문에서는 **이전에 사용된 네트워크보다 실질적으로 더 깊은 네트워크의 훈련을 용이하게 하기 위한 Residual learning 프레임워크(ResNet)를 제시한다.**
* Residual learning은 Residual function을 Leraing에 사용하는 것으로 layer를 재구성한다.
* 본 논문은 Residual 네트워크가 최적화하기 더 쉽고, 깊이가 증가한 상태에서 높은 정확도를 얻을 수 있다는 것을 보여준다.
* ImageNet 데이터 세트로 VGG 네트워크보다 8배 깊지만 낮은 복잡성을 가진 최대 152개의 레이어의 Residual 네트워크를 평가한다.
* 이러한 Residual 네트워크의 앙상블은 ImageNet 테스트 세트에서 3.57%의 오류를 달성한다.

# 1. Introduction
* Deep CNN은 이미지 분류분야에서 많이 사용되고 있으며, 이 네트워크의 깊이가 매우 중요하다는 사실들이 이전 연구들로 밝혀져왔으며 "ImageNet" 챌린지에서도 깊이가 매우 깊은 모델들을 이용하고 있다.
* 하지만 깊이가 깊어질수록 장애물이 생기는데, 그것은 **Vanishing/Exploding gradients**이라는 문제이다.
* 이러한 문제는 **Normalized initialization, intermediate normalization layer**로 대부분 해결되었으며, 이는 수십 개의 layer를 가진 네트워크가 역전파(Backpropagation)와 함께 확률적 경사 하강(SGD)를 위한 수렴을 가능하게 만들었다.

---
* 더 깊은 네트워크가 수렴을 시작할 수 있을 때, Degradation problem가 생긴다.
  * **Degradation problem**
  -> 네트워크 깊이가 증가함에 따라 정확도가 포화되었다가 급속하게 저하되는 문제

![](https://velog.velcdn.com/images/aioptlab/post/840868ab-525d-45b5-aa26-b8d98c511349/image.png)

* Figure 1을 보면 56-layer가 20-layer보다 Train-error, Test-error 둘 다 높은 것을 알 수 있다.
* 이러한 훈련 정확도의 저하는 모든 시스템이 최적화하기 쉬운 것이 아님을 나타낸다.
---
* 먼저 간단하게 Identity mapping layer를 추가했지만 좋은 솔루션은 아니었기 때문에 논문에서는 Deep residual learning 프레임워크라는 개념을 도입했다.
* 기존의 바로 Mapping하는 것을 H(x)라고 나타낼때, 논문에서는 비선형 layer의 적합하도록 하는 **F(x) = H(x) - x (출력과 입력의 차이)** 를 제안한다.
* 핵심은 **F(x)가 0이 되는 것이 최적의 해이고, H(x)를 x로 mapping하는 것이 학습의 목표가 된다.**

![](https://velog.velcdn.com/images/aioptlab/post/e1d37e22-c274-440c-bf13-eae09eba9d97/image.png)

* 이전에는 Unreferenced mapping인 H(x)를 학습시켜야한다는 점 때문에 어려움이 있었는데, 이제는 H(x) = x라는 최적의 목표값이 사전의 pre-conditioning으로 제공되므로 Identity mapping인 F(x)에 대한 학습이 더 쉬워진다.
* 입력에서 출력으로 바로 연결되는 것을 **Shortcut** 이라고 하는데 이는 "+x"를 의미하므로, 덧셈이 늘어난 것을 빼면 연산량의 증가는 없다. 
* 전체 네트워크는 역전파를 사용하여 SGD에 의해 종단 간으로 훈련될 수 있으며, Solver를 수정하지 않고도 쉽게 구현이 가능하다.

---
* 위 내용을 가지고 실험을 진행하는데 이때의 목적은 다음과 같다.
  * **Plain net(단순히 층을 쌓는)은 깊이가 증가할 때 더 높은 Train error를 갖는데, 논문의 모델은 그렇지 않다는 것을 보여준다.**
  * 더 깊어진 깊이에서 더 쉽게 높은 정확도를 얻을 수 있어 **이전 네트워크보다 더 나은 결과를 얻을 수 있음을 보여준다.**
---

# 2. Related Work
## 2.1 Residual Representations
* VLAD[1], Fisher Vector[2]의 사례를 들며, 벡터 양자화의 경우 Residaul vector encoding이 Original vector encoding보다 더 효과적인 것을 보여주고 있다.

> **벡터 양자화**
N개의 특징 벡터 집합 x를 K개의 특징 벡터들의 집합 Y로 Mapping하는것

* Low-level의 비전 및 컴퓨터 그래픽에서 Partial Differential Equations(PDEs)를 해결하기 위해 Multigrid[3] 방법이 사용된다.

> **Multigird 방식**
시스템을 여러 Scale의 하위 문제로 재구성하는 것이며, 여기서 각 하위 문제는 큰 Scale과 작은 Scale 간의 Residual을 담당한다.

* Multigird의 대안으로 계층 기반 pre-conditioning[4, 5]이 존재하며, 이 방식은 기존 방식보다 훨씬 빨리 수렴하는 특징이 있다. 이는 **최적화를 더 간단하게 수행해준다**는 것을 의미한다.

> **계층 기반 pre-conditioning (hierarchical basis pre-conditioning)**
두 Scale 사이의 Residual vector를 나타내는 변수에 의존하는 방식

## 2.2 Shortcut Connections
* Shortcut connections는 오랫동안 연구되어진 분야이며 초기에는 네트워크 입력에서 출력으로 연결된 선형 레이어를 추가하는 것이었다.
* Highway networks[6]라는 연결도 존재하는데, 이 연결 방식에는 Gating functions이 존재한다. 이러한 Gates는 Data-dependent하며 파라미터를 가진다.

> **Highway networks**
어떤 레이어를 통과할 때, 해당 레이어에서 수행되어야 하는 선형 연산과 활성화(Activation)과 같은 연산을 거치지않고, 빠르게 지나만 가는 우회로도 제공하자는 것
(참고자료 : https://brunch.co.kr/@kakao-it/142)

* ResNet의 Shortcut connection은 **파라미터가 추가되지 않으며, 모든 정보는 항상 통과해야하며 추가적인 Residual function을 학습하는 것이 가능하다.**

# 3. Deep Residual Learning
## 3.1 Residual Learning
* Introduction에 나왔던 내용과 비슷하다.
* H(x)를 기본 Mapping으로 간주하고, 여러 비선형 계층이 복잡한 함수를 점근적으로 근사할 수 있다고 가정한다면, H(x) - x의 Residual function을 점근적으로 근사할수 있다고 가정하는 것과 동등하다.
* 따라서, H(x) = F(x) + x으로 이항을 해 사용하게 되는데, 이는 학습에 용이해서라고 한다.
* 실제로는 Identity mapping이 최적일 가능성이 낮지만, 재구성 방식은 문제를 pre-condition하는데 도움을 준다.
* 따라서 pre-conditioning으로 인해 Optimal function이 zero mapping보다 identity mapping에 더 가깝다면, solver가 identity mapping을 참조하여 작은 변화를 학습하는 것이 새로운 function을 학습하는 것보다 더 쉬울 것이라고 설명한다.

## 3.2 Identity Mapping by Shortcuts

![](https://velog.velcdn.com/images/aioptlab/post/b7358a6f-197f-45aa-9be2-92f5bc720c35/image.png)

* 1번 식에서 **x와 y는 고려된 layer의 입력과 출력 벡터**이며, **F(x, {Wi})는 학습할 Residual mapping**을 나타낸다.
* **F = W2θ(W1x)** 로서, θ는 ReLU를 나타내고, 편향을 생략하여 주석을 간략화하였으며, **F + x** 는 Shortcut connection과 Element-wise addition에 의해 수행된다.
* F + x 연산을 위해서는 F와 x의 차원이 같아야하는데, 이들이 서로 다를 경우 2번 식처럼 Linear projection인 Ws를 곱하여 차원을 갖게 만들어준다.

![](https://velog.velcdn.com/images/aioptlab/post/4ae3c266-84c4-4c49-9313-e97e3e2c84b5/image.png)

## 3.3 Network Architectures
![](https://velog.velcdn.com/images/aioptlab/post/b5381a7a-308c-4a0d-b25a-58e07461aebd/image.png)

### 3.3.1 Plain Network
* Baseline 모델로 사용한 Plain 네트워크(Figure 3의 34-layer plain)는 VGGNet에서 영감을 받아 구성하였으며, 아래 2가지 규칙을 가지고 설계했다.

1. **동일한 출력 Feature map 크기에 대해 레이어는 동일한 수의 필터를 가지고 있다.**
2. **Feature map 크기를 절반으로 줄이면 레이어당 시간 복잡성을 보존하기 위해 필터 수가 두 배로 증가한다.**

* Convolution layer는 3x3의 필터, stride = 2로 Downsampling, Global average pooling layer를 사용했고, 마지막에는 softmax를 사용한 1000-way FC layer를 통과시켰다.
* 전체 layer의 수는 34개인데, 이는 VGGNet보다 적은 필터와 복잡성을 가진다.

### 3.3.2 Residual Network
* Residual Network는 Plain 모델에 기반하여 Short connection을 추가해 구성하였다.
* 입력과 출력의 차원이 같다면, Identity shortcut을 바로 사용하면 되지만, Dimension이 증가한다면 두가지 선택지가 주어진다.

1. **Zero padding을 통해 차원을 키워준다.**
2. **2번 식에서 사용한 방법인 Linear projection을 사용한다.**

* 두 옵션 모두에서 Shortcut connection이 Feature map을 두칸씩 건너뛰므로 stride를 2로 설정한다.

## 3.4 Implementation
* 모델 구현은 아래와 같다.

1. 짧은 쪽이 [256, 480] 사이가 되도록 무작위로 샘플링된 이미지의 크기가 조정된다.
2. Horizontal flip 부분적으로 적용 및 per-pixel mean을 빼준다.
3. 224x224 사이즈로 무작위로 Crop 수행
4. Standard color augmentation을 적용한다.
5. Batch Normalization 적용
6. 가중치 초기화 후 처음부터 훈련 진행
7. Mini-batch size가 256인 SGD 사용
8. Learning rate는 0.1에서 시작하며 정체 시 10씩 나눈다.
9. Weight decay는 0.0001, Momentum은 0.9
10. 최대 64 X 10^4번 반복 수행하며 Dropout은 사용하지 않는다.

* 테스트 단계에서는 10-cross validation 방식을 적용하고, 이미지는 더 짧은 쪽이 {224, 256, 384, 480, 640}이 되도록 크기를 조정한 뒤, Average score를 구한다.

# 4. Experiments
* 1000개의 클래스로 구성된 ImageNet 2012 분류 데이터 세트에서 ResNet을 평가한다. 
* 모델은 128만 개의 훈련 이미지에 대해 훈련되고 50k 유효성 검사 이미지에 대해 평가된다.
* 100k 테스트 이미지에 대한 최종 결과를 얻는다.

## 4.1 ImageNet Classification


![](https://velog.velcdn.com/images/aioptlab/post/07f5488c-92d2-4817-95f1-75189d89e8a0/image.png)

* 각 모델 구조의 세부적인 내용은 Table 1을 확인하면 알 수 있다.

![](https://velog.velcdn.com/images/aioptlab/post/1ef78420-cf04-4b20-b936-bc4c71bf1f04/image.png)

![](https://velog.velcdn.com/images/aioptlab/post/7493d542-d5f1-4ce3-9f51-5efae495434c/image.png)

---
### Plain Network
* Figure 4의 왼쪽을 보면 34-layer Plain 네트워크가 18-layer Plain 네트워크보다 더 깊은데도 불구하고, 더 높은 훈련 오류를 가지고 있다. **(Degradation 문제 발생)**
* 저자들은 이러한 최적화 어려움이 Vanishing gradient 때문에 발생하는 것은 아니라 판단했는데, 이는 Batch Normalization 등이 적용되어 순전파, 역전파 신호가 사라지지 않았기 때문이라고 한다.
* 실제로 34-layer Plain 네트워크는 경쟁력있는 정확도를 달성할 수 있었으며, 저자들은 Deep plain 네트워크는 기하급수적으로 낮은 수렴률을 가지기 때문에 Train error의 감소에 영향을 끼쳤을 것이라 추측했다.
---
### ResNet
* Residual learning에서는 34-layer ResNet이 18-layer ResNet보다 2.8% 좋은 성능을 보였으며, 34-layer ResNet에서 낮은 Train error를 보였다. **(Degradation 문제 해결)**
* **이는 Depth가 증가했을 때 더 좋은 정확도를 얻을 수 있음을 의미한다.**
* 34-layer ResNet의 Top-1 error는 3.5%가량 줄었고, 이는 Residual learning이 Extremely deep system에서 매우 효과적임을 알 수 있다.
---
* **18-layer ResNet과 Plain net은 비교적 정확**하지만, 18-layer의 ResNet이 더 빨리 수렴하였다. 즉, 모델이 과도하게 깊지 않은 경우 (18-layer), 현재의 SGD Solver는 여전히 Plain net에서도 좋은 solution을 찾을 수 있지만, **ResNet은 같은 상황에서 더 빨리 수렴할 수 있다.**

### Identity vs. Projection Shortcuts
* 앞에서 파라미터가 없는 Identity shortcut이 학습에 도움이 된다는 것을 보여주었다. 다음으로는 Projection shortcut을 조사를 위해 3가지 옵션을 비교한다.
---
**Option**

A. Zero padding shortcut을 사용한 경우 (Tabel 2, Fig 4의 오른쪽과 동일)
B. 차원 증가에 Projection shortcut을 사용한 경우
C. 모든 shortcut으로 Projection shortcut을 사용한 경우

---
![](https://velog.velcdn.com/images/aioptlab/post/f61667cf-8fd7-402d-8e80-20839f71ff6d/image.png)

* 3가지 옵션 모두 Plain model보다 좋은 성능을 보였고, 그 순서는 A < B < C이나 3가지의 성능에는 큰 차이가 없었으므로, **Projection shortcut이 Degradation 문제를 해결하는데 필수적이지 않다는 것을 확인했다.**
* 따라서, 논문에서는 메모리/시간 복잡성 및 모델 크기를 줄이기 위해 C 옵션을 사용하지 않았다.
* Identity shortcut은 아래 병목 구조들의 복잡성을 증가시키지 않기 위해 특히 중요하다.

### Deeper Bottleneck Architectures
![](https://velog.velcdn.com/images/aioptlab/post/91033692-f3ed-494b-8351-b6def7c8c510/image.png)

* ImageNet 학습을 진행할 때, 훈련 시간이 길어질 것을 대비하여 Building block을 Bottleneck design으로 수정했다.
* Figure 5의 오른쪽처럼 사용하였으며, 1x1은 차원을 줄이거나 늘리는데 사용되어 3x3 layer의 입력과 출력 차원을 줄인 **Bottleneck 구조**를 만들어준다.

* 파라미터가 없는 Identity shortcut은 Bottleneck 구조에서 특히 중요한데, 만약 Identity shortcut이 Projection shortcut으로 대체되면, 시간복잡도와 모델 사이즈가 2배로 늘어난다.
* 따라서, Identity shortcut은 Bottleneck 구조를 더 효율적인 모델로 만들어준다.

#### 50-layer ResNet
* 34-layer ResNet의 2-layer block을 3-layer bottleneck block으로 대체하여 50-layer ResNet을 구성하였다. 이때, dimension matching을 위해 B 옵션을 사용하였다.

#### 101-layer and 152-layer ResNets
![](https://velog.velcdn.com/images/aioptlab/post/2a62c84c-1bc6-4b1b-908e-660446996067/image.png)

* 더 많은 3-layer block을 사용하여 101-layer 및 152-layer ResNet을 구성하였다. 깊이가 상당히 깊어졌음에도 불구하고 VGG-16 / 19 모델보다 낮은 복잡성을 가졌다.
* Degradation 문제없이 상당히 높은 정확도를 보였다. (Table 3, 4)

### Comparisons with State-of-the-art Methods
* Table 4에서는 이전의 SOTA 모델의 결과와 ResNet을 비교하고 있다. ResNet-34는 경쟁력있는 정확도를 달성했다.

![](https://velog.velcdn.com/images/aioptlab/post/20dae3f4-0388-47da-8f08-645c0fcb4013/image.png)

* Table 5를 보면, 이 단일 모델 결과는 이전의 모든 앙상블된 모델들의 결과를 능가하며 Top-5 error가 3.57% 발생하며 ILSVRC 2015에서 1위를 차지했다.

## 4.2 CIFAR-10 and Analysis
* 저자들은 ResNet을 활용하여 0개의 클래스에서 50k개의 훈련 이미지와 10k개의 테스트 이미지로 구성된 CIFAR-10 데이터 세트에 대해 실험한 결과를 보여준다.

![](https://velog.velcdn.com/images/aioptlab/post/7180d894-36bd-4230-957c-ca84d8299c62/image.png)

* **Plain/Residual 모델은 Figure 3에 나와있는 것을 사용**했으며, 첫번째 Layer를 3x3 Convolution으로 구성한 뒤, 위 표처럼 모델을 구성했으며, 말단에 10-way FC layer를 붙여서 총 **6n + 2개의 Layer가 존재하는 구조**이다.
* Shortcut connection을 사용할 때는 Identity shortcut을 사용한다. (Option A)

![](https://velog.velcdn.com/images/aioptlab/post/180f46c5-e106-4b47-a558-a699e2ded317/image.png)

* Figure 6의 왼쪽은 Plain 네트워크의 동작을 보여주는데, 깊이가 증가할수록 높은 Train error를 가지는 것을 알 수 있다.
* Figure 6의 가운데는 ResNet의 동작을 보여주는데, ResNet은 최적화 어려움을 극복하고 깊이가 증가할 때 정확도 향상을 보여준다.

![](https://velog.velcdn.com/images/aioptlab/post/39dc315f-4be3-46f6-b3bd-97113050e362/image.png)

* Figure 6의 가운데의 ResNet-110이 잘 수렴되는 것을 볼 수 있는데, 이를 FitNet, Highway와 같은 다른 네트워크들과 비교하자면, 매개변수가 적지만 가장 좋은 결과를 얻는 것을 알 수 있다.

### Analysis of Layer Responses
![](https://velog.velcdn.com/images/aioptlab/post/d1e92d20-b428-4f82-a784-6b24c36cd22d/image.png)

* Figure 7은 Layer Responses의 표준편차를 나타내는데, ResNet이 Plain 네트워크보다 더 작은 Responses를 가지고 있음을 보여준다.
* 이는 Residual function이 Non-residual function보다 일반적으로 0에 가까울 것이라는 주장을 뒷받침한다.

### Exploring Over 1000 layers
* Figure 6의 오른쪽과 Table 6의 ResNet-1202를 보게되면, ResNet-110보다 결과가 좋지 않지만, 둘 다 유사한 Train error를 가지고 있다.
* 저자들은 이를 Overfitting 때문이라고 주장한다.

## 4.3 Object Detection on PASCAL and MS COCO
![](https://velog.velcdn.com/images/aioptlab/post/c869a070-bcb3-455b-9f36-4198763a423d/image.png)

* ResNet이 다른 Recognition 작업에서 우수한 일반화 성능을 가지고 있음을 Table 7, 8로 보여주면서 뛰어난 성능을 자랑하고 있다. 자세한 내용은 논문 부록에 존재한다.

# Reference
[1] H. Jegou, F. Perronnin, M. Douze, J. Sanchez, P. Perez, and C. Schmid. "Aggregating local image descriptors into compact codes". TPAMI, 2012.

[2] F. Perronnin and C. Dance. "Fisher kernels on visual vocabularies for image categorization". In CVPR, 2007.

[3] W. L. Briggs, S. F. McCormick, et al. "A Multigrid Tutorial". Siam, 2000.

[4] R. Szeliski. "Fast surface interpolation using hierarchical basis functions". TPAMI, 1990.

[5] R. Szeliski. "Locally adapted hierarchical basis preconditioning". In SIGGRAPH, 2006.

[6] R. K. Srivastava, K. Greff, and J. Schmidhuber. "Highway networks". arXiv:1505.00387, 2015.
