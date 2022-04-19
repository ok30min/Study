이 글은 논문 **DCPR-GAN: Dental Crown Prosthesis Restoration Using Two-Stage Generative Adversarial Networks (2022)** 에 대한 설명입니다. 논문 원본에 대한 링크는 아래에 적어놓았습니다.

논문 원본 : https://ieeexplore.ieee.org/abstract/document/9568708

# Abstract
* 데이터 중심 관점에서 Dental crown surface을 재구성하기 위해 새로운 2단계 **Generative Adversarial Network (GAN)** 를 설계하여 이 문제를 해결한다.

* 첫 번째 단계에서, 결함이 있는 치아와 대상 크라운 사이의 고유한 관계를 학습하기 위해 Conditional GAN(CGAN)이 설계되어 교합 관계 복원 문제를 해결할 수 있다.

* 두 번째 단계에서는 Occlusal groove parsing network(GroNet)와 Occlusal fingerprint 제약 조건을 고려하는 Generator를 사용하여 Occlusal surface의 기능 특성을 풍부하게 하여 개선된 CGAN이 추가로 고안된다.

* 결과는 제안된 프레임워크가 실제 환자 데이터베이스를 사용한 Occlusal surface 재구성에서 최첨단 딥러닝 방법을 크게 능가한다는 것을 보여준다.

# Introduction
* 치아의 임상 실습에서 치아의 형태학적 다양성과 치아의 특정한 Occlusal fingerprints를 포함하는 치아를 재구성하는 것은 정말 어려운 일이다

![](https://velog.velcdn.com/images/aioptlab/post/04140efc-7e19-4bde-b5ad-92c807a437fd/image.png)

* 3Shape, Duret 및 OrthoCAD와 같은 대부분의 Computer-Aided Geometric Design(CAGD) 기반 Dental crown prosthesis(DCP) 복원 시스템은 구강 보철 소프트웨어의 중요한 부분으로 **표준 치아 템플릿 라이브러리**를 사용한다.

* CAGD와 보철치과의 결합은 많은 이점을 가져오지만, DCP의 안면상 형태학적 특성은 치과 의사의 숙련도에 따라 달라진다.

* 가장 적합한 교합 표면은 무엇이며, 얼마나 많은 특징점을 선택해야 하며, 32개 범주의 치아에 대한 교합 기능 영역을 정량화하는 방법과 같은 결정이 필요하다.

* 이에 치과 의사의 업무량을 덜어주는 것은 물론 치아 복구 비용 절감을 위한 데이터 중심 DCP 복원 개발이 시급하다.

* **Deep learning 기반으로 DCP 프레임워크를 개발하는 것은 어렵다.**
1) 데이터 부족
2) 치아의 크기와 모양이 매번 달라서 DNN 모델 훈련의 어려움을 증가시킨다.
3) 반대쪽 치아에 대한 고려가 어려움

* **이 논문에서 제안하는 DCPR-GAN의 장점**
1) 서로 다른 범주의 치아를 복원하는데 적합하다.
2) 기존의 복원 방식이 환자의 치아에 적합하기 어렵다는 문제를 피할 수 있다.
3) Occlusal surface를 정확하게 표현하는 보형물을 설계하는데 효율적이다.

# Method
* 제안하는 DCPR-GAN의 흐름도는 아래 그림과 같다.

![](https://velog.velcdn.com/images/aioptlab/post/1774fc27-96fd-4c96-82d6-ee36760fcc78/image.png)

* Dental occlasul surface reconstruction network(복원 네트워크)는 Dental crown image를 재구성하도록 설계되었으며, 여기서 생성된 Occlasul surface의 품질을 향상시키기 위해 2단계의 GAN이 개발되었다.

* **Stage-I GAN**은 결함이 있는 치아와 조직 구조 사이의 고유한 연관성을 고려한다.
* **Stage-II GAN**은 Occlasul fingerprint, Groove를 고려하여 미세 조직 구조를 재구성한다.

## 1) Depth Map Generation
* 먼저 치아의 형태를 담은 데이터를 만들기 위해 모든 공간 정보를 담아 아래 그림과 같은 방법으로 Depth Map으로 표현한다.

![](https://velog.velcdn.com/images/aioptlab/post/1b8cd4de-8b37-4ff1-9db5-7a003f5b0129/image.png)

* 먼저 Crown 부분과 평행한 투영면(256x256)을 위치시킨다. 그 다음 Crown 표면에 대한 투영 평면의 위치(i,j)에서 Crown까지의 최단 거리 dij를 구한다.
(좌표의 원점은 가장 멀리 떨어진 평면의 왼쪽 하단 모서리에 설정된다.)
* 가장 멀리 떨어진 **Crown 너머의 값들은 모두 0**으로 설정한다.

* 마지막으로, 각 격자의 중심점에서 Occlusal surface까지의 **최단 수직 거리 d**를 고려하여 아래 식과 같이 깊이에 대한 값을 p(i,j)로 변환한다.
(**MaxI는 8bit일 경우 255를 나타내고, n은 이미지 향상 계수, l은 거리 임계값을 나타낸다.**)

![](https://velog.velcdn.com/images/aioptlab/post/07a50634-c599-499f-ba1c-52086539ea13/image.png)

* 실험을 통해 **n=2, l=6mm**일 때, Occlusal surface의 기능적 특성 정보를 유지할 수 있었다.

## 2) Two-Stage Dental Crown Restoration Network

* **Stage-I에서는 공간적 위치 관계를 만족하는 Occlusal surface의 기본 형상을 만들고, Stage-II에서는 Occlusal groove parsing network(GroNet) 손실과 Occlusal fingerprints 제약 조건을 추가하여 Occlusal surface의 세부 사항을 완성한다.**

* **GroNet은 Stage-I GAN 모델과 그 매개 변수에 의해 사전 훈련된다.** 네트워크가 수렴된 후, 생성된 Occlusal groove와 Object groove 픽셀의 조화를 더욱 향상시키기 위한 제약조건으로 Stage-II 네트워크에 고정되고 로드된다.

* 그림 4는 두 개의 GAN으로 구성된 제안된 DCPR-GAN 아키텍처를 보여준다.

![](https://velog.velcdn.com/images/aioptlab/post/ac264440-30f3-4a8a-b039-0e2921d3b095/image.png)


### 1. Initial Occlusal Surface Generation (Stage-I GAN)
* GAN의 목표는 기존 데이터의 잠재적 분포를 모델링하고 동일한 분포를 가진 새로운 데이터 샘플을 생성하는 것이다.
* 이 논문에서는 **목적 함수 V(D, G)** 를 가진 기존의 GAN에 **추가 조건 변수 c**를 사용한다.

![](https://velog.velcdn.com/images/aioptlab/post/9b94ed50-6a3e-4a61-8e4f-48f687735ffb/image.png)

	-> D : Discriminator
	-> G : Generator
	-> x : Real image / Pdata(x) : Real data distribution
	-> z : input sample (noise image) / Pz(z) : input sample distribution

* Generator는 노이즈가 많은 샘플 z를 입력으로 받아들이고, 실제 훈련 샘플 x에 대한 복잡한 매핑 관계를 학습하며, 분포 Pz(z)를 실제 데이터 분포 Pdata(x)에 매핑하려고 시도한다. 즉, **G는 G(z|c)와 x 사이의 분포 거리를 최소화하는 것을 목표로 한다.**

* Discriminator는 생성된 이미지 G(z|c)와 실제 이미지 x를 구별하기 위해 이진 분류기로 사용된다. 즉, **D의 목표는 Pz(z)와 Pdata(x) 차이를 최대화하는 것이다.**

* Occlusion spatial relationship(폐색 공간 관계)에 초점을 맞춘 첫 번째 단계에서 기본 형태를 생성하는 것이 전략이다.

* 아래는 Stage-I의 Generator의 Loss function이다.

![](https://velog.velcdn.com/images/aioptlab/post/62ad058d-0671-478c-a92c-e0f577f942c0/image.png)

* 아래는 Stage-I의 Discriminator의 Loss function이다.

![](https://velog.velcdn.com/images/aioptlab/post/7cc2da42-3e21-45f3-879f-09ebdf77d0d1/image.png)

* **Occlusal Surface의 해부학적 특징과 공간적 위치에만 초점을 맞추면 결함이 있는 치아의 완전한 기능을 복원할 수 없다. 또한, 씹는 운동 중에 윗니와 아랫니 사이의 접촉을 고려해야 한다.**

* (5)는 Occlusal Surface 관계를 측정하기 위한 제약조건으로서 **preparation tooth인 x1, opposing tooth인 c1, tooth type label인 ĉ, Occlusal fingerprint z1이 없는 Target crown, 두 턱 사이의 간격 거리 d**를 사용하여, G1이 올바른 Occlusal 관계를 가지도록 유도한다.

![](https://velog.velcdn.com/images/aioptlab/post/ec1e60fb-32f7-4b47-93a3-44721d480d1a/image.png)

* D1의 Hidden layer에서 생성된 Occulsal surface와 Target crown 사이의 high-dimensional feature deviation가 측정된다.

* Adversarial process를 통해 D1은 그들 사이의 불일치를 최대한 포착할 수 있다. 반대로, G1은 생성된 폐색 표면을 대상 크라운에 가깝게 적용하려고 시도합니다.

* Perceptual generation loss는 아래와 같다.

![](https://velog.velcdn.com/images/aioptlab/post/cbc2ab10-1f00-4823-b199-7aeddfa35f73/image.png)

	-> Ci×Hi×Wi는 i번째 Hidden layer인 hi의 형태

* Perceptual adversarial loss는 아래와 같다.

![](https://velog.velcdn.com/images/aioptlab/post/f8289c26-822c-4237-863c-fd1cf361f893/image.png)

	-> m : positive margin value


### 2. Functional Occlusal Surface Generation (Stage-II GAN)
* **Stage-I의 Generator인 G1은 Pose 및 Basic shape이 Target crown과 유사한 거친 Occlusal surface를 생성한다.** 따라서 우리는 두 번째 단계의 Generator인 G2를 사용하여 Initial Crown을 Target에 더 가깝게 적용하는 세밀한 Occlusal surface features을 생성한다.

* The new generation loss는 (8)과 같다.

![](https://velog.velcdn.com/images/aioptlab/post/bbde8ba7-ef16-46fd-a070-5a0007d96f31/image.png)

* The adversarial loss는 (9)과 같다.

![](https://velog.velcdn.com/images/aioptlab/post/87d0790c-46b3-4368-b8ef-cc07c3842115/image.png)

* Stage-II의 Conditional adversarial loss는 (10)과 같다.

![](https://velog.velcdn.com/images/aioptlab/post/022d858f-78d6-4c25-ba53-12c094eca361/image.png)

* Occlusal Surface의 미세한 조직 구조를 재구성하는 것을 목표로 새로운 GroNet을 도입하여 생성된 표면을 더욱 현실적인 Crown features를 보장한다.

* Parsing network에서, 두 Occlusal grooves 사이의 불일치는 L1 regularization에 의해 최소화되며, 제안된 Groove loss은 다음과 같이 정의된다.

![](https://velog.velcdn.com/images/aioptlab/post/a741c253-7ce1-4710-8af0-8192f820aa1c/image.png)

	-> ||.|| : L1 norm
	-> z : Target tooth
	-> c : The opposite occlusal tooth
	-> x : Noise image
    -> F(.)는 생성된 표면의 Occlusal groove과 해당 Target crown을 추출하는 데 사용되는 Occlusal network를 나타낸다.


## 3) Light-Weight Intermediate Connetor Design

![](https://velog.velcdn.com/images/aioptlab/post/39dd7a71-912f-4253-a8ad-b5a42652e89c/image.png)

* (a)는 Tooth preparation의 상부를 일정한 거리만큼 오프셋하여 접착층의 사용을 시뮬레이션하는 것이다.

* (b)는 생성된 Occlusal surface(SecL1)과 접착층(SecL2)의 Boundary curves을 기준선으로 하여 Connector mesh surface을 설계하는 것이다.

* Connector surface에 대한 형상 제어를 개선하기 위해 B-spline interpolation을 기반으로 한 Skinning 작업을 사용하여 중간 커넥터를 자동으로 설계합니다. 여기서 Skinning은 CAD 시스템에서 표면 정의의 일반적인 방법이다.

* Connector 설계 과정
1) The boundary curve SecL2의 모든 점을 추출한다. **SecQ = {qi|i = 1, 2, …, n}**
2) 기준점 시퀀스로 SecQ를 사용하고, Boundary curve SecL1에 해당하는 Matching point 시퀀스 SecP를 계산하기 위해 "Plane intersection method"을 채택한다. **SecP = {pi|i = 1, 2, …, n}**
3) 기준점 시퀀스에 대한 두 세트의 중간점 시퀀스 SecK를 계산, **ki = (pi + qi)/2, i = 1, 2, …, n;**
4) (pi, ki, qi)를 Control point 집합으로 하여 B-spline 곡선을 이용하여 능선을 구한 후 균일하게 이산화한다.
5) 인접한 두 능선 사이의 교차점을 순차적으로 연결하여 삼각형 Mesh surface를 얻는다.

# Experiments and Results
## 1) Dental Dataset Preparation
* 데이터는 베이징 대학병원과 난징 병원에서 치과의사들이 추출한 Occlusal fingerprint를 사용했으며, 치과의 스캐너는 2가지를 사용했다.

![](https://velog.velcdn.com/images/aioptlab/post/74684d78-ef93-49b4-95aa-da67d4399b2f/image.png)

* **총 데이터의 개수는 780개**이며 모든 샘플은 아래 그림과 같이 구성되어져있다.

![](https://velog.velcdn.com/images/aioptlab/post/da34e50a-10cb-4119-83b1-6561a7f636c9/image.png)

## 2) Training Details
* **700개의 Data를 Train set**으로 사용했고, **80개의 Data를 Test set**으로 사용했으며, 학습시킬 때 사용했던 Parameter는 다음과 같다.

![](https://velog.velcdn.com/images/aioptlab/post/b1e75f55-db54-48c0-8e0f-1ef91d472663/image.png)

* 아래 그래프는 Training 시 Loss를 나타내며 학습이 진행됨에 따라 점점 감소하는 것을 알 수 있다.

![](https://velog.velcdn.com/images/aioptlab/post/d389c436-6707-43f9-8225-0fdc4c641dc4/image.png)


## 3) Effectiveness of Two-Stage Generator Network
* Stage-I GAN은 Guiding 조건으로서 Occlusal fingerprint의 효과를 검증하기 위해 훈련된다.
* 아래 그림 같이 생성된 치아 패턴의 분포(Fingerprint-Output)는 목표 위치(Fingerprint-Object)에 매우 근접하다.
* Stage-I GAN은 뾰족한 부분과 Groove와 같은 Crown의 전체 구조를 포착해 Occlusal surface를 만든다. 그러나 **세부 정보들이 누락되어 거친 표면이 나온다.**

![](https://velog.velcdn.com/images/aioptlab/post/4c87a2b2-9f5f-482a-9874-deeac96833a6/image.png)

* Stage-II GAN에 의 생성된 Occlusal surface(Crown-Stage-II)는 Dental cusp-fossa의 분포가 뚜렷하다. 이는 **치아의 기능적 특성을 더 잘 반영**한다는 것을 보여준다.
* 마지막으로, DCPR-GAN의 성능을 추가로 검증하고 Occlusal groove shape 제약이 Generator에 미치는 영향을 분석하기 위해, GroNet을 사용하여 Stage-I와 Stage-II 단계의 Occlusal groove를 추출한다.
* Groove-Stage-II와 Groove-Object가 매우 유사한 것을 알 수 있다.

## 4) Comparison Results with SOTA Methods
### 1. Quantitative Results
* **Pix2pix [1], Pix2pixHD [2], perceptual adversarial network(PAN) [3], generative face completion (GFC) network [4], and dental occlusal surface generator network (Dental-GAN) [5]**를 비교한다.

![](https://velog.velcdn.com/images/aioptlab/post/5b4f07e7-094b-41e2-adec-21b1f34b13b9/image.png)

	-> PSNR : Peak Signal Noise Ratio
    -> RMSE : Root Mean Square Error
    -> SSIM : Structural Similarity Index Measure
    -> FSIM : Feature Similarity Index Measure

* Pix2pix는 PSNR, RMSE, SSIM 및 FSIM에 대한 값이 낮으므로, 폐쇄 표면의 더 자세한 정보를 재구성할 수 없음을 나타낸다. Pix2pix와 비교했을 때, 다른 7가지 방법은 더 나은 성능을 제공한다.

* **DCPR-GAN은 제안된 Two-stage generative network와 GroNet Parser의 효과를 검증하면서 네 가지 품질 지표에 대해 전반적으로 최고의 결과를 달성한다.**

![](https://velog.velcdn.com/images/aioptlab/post/76a8fb69-7381-4d5c-b20b-6336870fc2ac/image.png)

* 원본 Image와 재구성된 Image 간의 편차 측정의 유사성을 평가하기 위해 일원 분산 분석(ANOVA) 테스트를 수행했고, 그 결과, 8가지 방법의 경우 편차 측정에서 통계적으로 유의한 차이를 발견했다(p < 1e-8).

* Kruskal-Wallis 검정을 사용하여, 우리는 DCPR-GAN과 다른 7가지 방법(p ≤ 1e-3) 사이에 통계적으로 유의한 차이를 발견했다.


### 2. Qualitative Results
* 그림 10에는 치과 전문의에 의해 Occlusal fingerprints(Olive-drab color)이 추출되는 상기 8가지 방법으로 재구성된 대표적인 3가지 예가 제시되어 있다. 또한, 이 세 가지 예는 서로 다른 연령대의 다른 결함 있는 치아를 가진 환자들(#36 또는 #46)에서 선택된다.
* 첫 번째 열은 세 개의 준비 치아 샘플(Dark seagreen color)을 제공하고, 두 번째 열은 상응하는 Ground-truth samples을 제공합니다.

![](https://velog.velcdn.com/images/aioptlab/post/82a7e3b7-4258-49d1-9be4-5b1efad85c4d/image.png)

* Pix2pix, Pix2pixHD 및 PAN에 의해 생성된 Occlusal surface에 Occlusal fingerprints이 적거나 Occlusal groove가 더 매끄러운 것을 나타낸다.

* GFC, Dental-GAN, Stage-I GAN은 Pix2pixHD, PAN보다 성능이 뛰어나다.

* Stage-I GroNet_OF에 의해 생성된 Occlusal fingerprints과 Occlusal groove의 분포는 다른 방법보다 더 합리적이다. **DCPR-GAN의 결과는 특히 Occlusal fingerprints 분포와 Occlusal surface의 형태학적 특성에 대해 실측 샘플에 비교적 가깝다.**

![](https://velog.velcdn.com/images/aioptlab/post/c13058a4-cf4d-4945-995a-56192034e0ec/image.png)

* Occlusal surface의 품질을 평가하기 위해 생성된 Occlusal surface와 Target crown 사이의 편차는 동일한 근위 치아의 제약 하에서 계산된다. **그림 11에서 보는 바와 같이, 제안된 DCPR-GAN 방법은 다른 7가지 방법에 비해 훨씬 낮은 편차값을 달성한다.**

## 2) Realworld Dental Crown Prothesis
* 이 논문에서 제안된 접근 방식의 임상 적용을 입증하기 위해 부분적으로 치아를 가진 환자의 실제 Dental Crown 보철물 Data를 선택했다.

![](https://velog.velcdn.com/images/aioptlab/post/e49d2e4b-80c9-4e57-b694-f7ed10361eeb/image.png)

* 결손 치아의 Occlusal surface 이미지는 Region growing method을 사용하여 3D Crown 표면으로 재구성됩니다. 그런 다음, 계산된 접착층과 중간 커넥터를 결합하여 기능성 Occlusal 표면을 갖는 치과용 크라운을 설계한다.
* Occlusal surface는 움직임의 방향을 반영하여 **자연스러운 Occlusal movement를 재현했으며, 자연 치아의 해부학적 특징을 가지고 있어 기존 치아와 조화를 이루는 것을 알 수 있다.**

# Conclusion and Discussion
* 이 논문에서는 결함이 있는 Dental crown surface를 자동으로 재구성하는 새로운 2단계 치과 보철물 복원 프레임워크를 제안한다.
* 실제 데이터베이스를 기반으로 DCPR-GAN을 평가해본 결과 해당 알고리즘이 SOTA임을 보여준다.

* **향후 발전해야하는 점**
1) **충치 발명률이 높은 #36, #46 치아만을 고려**했기 때문에 다른 치아에 대해서는 더 많은 연구가 필요하다.
2) 직교 투영법은 깊이 정보만을 사용하므로 치열이 불규칙할 경우 치아의 깊이 정보가 불완전할 수 있다. 따라서 **다른 각도의 다중 깊이 맵을 추가로 고려**해야한다.


# References
[1] P. Isola, J.-Y. Zhu, T. Zhou and A. A. Efros, "Image-to-image translation with conditional adversarial networks", Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 1125-1134, 2017.

[2] T.-C. Wang, M.-Y. Liu, J.-Y. Zhu, A. Tao, J. Kautz and B. Catanzaro, "High-resolution image synthesis and semantic manipulation with conditional gans", IEEE Conf. Comput. Vis. Pattern Recognit., pp. 8798-8807, 2018.

[3] C. Wang, C. Xu, C. Wang and D. Tao, "Perceptual adversarial networks for image-to-image transformation", IEEE Trans. Image Process., vol. 27, no. 8, pp. 4066-4079, Aug. 2018.

[4] Y. Li, S. Liu, J. Yang and M.-H. Yang, "Generative face completion", Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 3911-3919, 2017.

[5] F. Yuan et al., "Personalized design technique for the dental occlusal surface based on conditional generative adversarial networks", Int. J. Numer. Methods Biomed. Eng., vol. 36, no. 5, 2020.
