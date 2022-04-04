이 글은 논문 **Robust Optimization for Process Scheduling Under Uncertainty(2008)** 에 대한 설명입니다. 논문 원본에 대한 링크는 아래에 적어놓았습니다.

논문 원본 : https://pubs.acs.org/doi/full/10.1021/ie071431u

# Abstract
* 이 논문은 Robust Optimization을 이용한 공정 스케줄링의 불확실성 문제를 다룬다.

* **Robust counterpart optimization은 기존의 시나리오 기반의 확률적 프로그래밍 방법에 비해, 불확실한 파라미터 수에 따라 문제의 규모가 증가하지 않는다는 이점을 갖는다.**

* **Soyster’s worst-case scenario formulation, Ben-Tal and Nemirovski’s formulation, and a formulation proposed by Bertsimas and Sim**라는 세가지 Robust counterpart optimization 공식을 연구하여 불확실한 스케줄링 문제를 풀었다.

* 결과적으로는 **Bertsimas and Sim**이 제안한 model이 불확실한 스케줄링 문제에 가장 적합한 모델이다.
  -> **Bertsimas and Sim**이 제안한 model의 이점
    1. Model은 다른 Formulations와 같은 size이다.
    2. 선형성을 보존한다.
    3. 모든 조건에 대한 conservatism의 정도를 제어할 수 있는 능력을 가지고 있고, Robust optimization problem에 대한 실현 가능성을 보장한다.

# 1. Introduction
* 불확실성을 갖는 Process scheduling에 대해 많은 현장에서 큰 관심을 가지고 있는데 이는 스케줄링과 관련된 많은 매개변수가 정확히 알려져 있지 않기 때문이다.
* 불확실성을 처리하는 스케줄링 방법은 Reactive scheduling과 Preventive scheduling으로 나뉜다. 이 논문에서는 Preventive scheduling에 대해 다룬다.
  - **Reactive scheduling** : 불확실성이 발생했을 때 원래 스케줄링 정책을 변경하거나 새롭게 생성하는 것
  - **Preventive scheduling** : 불확실성이 발생하기 전에 Robust한 스케줄링 정책을 생성하는 것

* Robust schedule을 생성할 때는 시나리오 기반 확률 프로그래밍과 Robust counterpart optimization이 있다.
* 과거 Robust scheduling 연구들은 시나리오 기반 formulation을 따르고 있으며, 이들은 불확실한 parameter의 수가 늘어날 때마다 시나리오 수가 기하급수적으로 증가하는 문제가 있다. 이 때문에 다수의 불확실한 parameter가 존재하는 문제를 해결할 때 어려움을 겪는다.

* 이러한 시나리오 기반 formulation의 대안으로 Robust counterpart optimization이 제안되었다. 
* Robust counterpart optimization의 주요 장점은 불확실한 데이터의 기본 확률 분포에 대한 가정이 필요하지 않으며, Risk에 대한 다른 Attitudes를 통합하는 방법을 제공한다는 것이다.

# 2. Robust Optimization
* Robust Optimization의 목적은 불확실한 데이터에 따라 다양하게 발생하는 결과에 잘 대처할 수 있는 Solution을 선택하는 것이다. 이때 불확실한 데이터는 알 수는 없지만 경계가 있는 것으로 가정되며, 그 공간의 Convexity를 가정한다.

* 불확실한 Parameter를 갖는 최적화 문제는 Robust counterpart optimization 문제로 재구성된다. 이때 불확실성 데이터의 확률 분포에 대한 정보를 요구하지 않고, Expected value objective function을 최적화하지 않는다. 단지, 주어진 불확실한 공간 전체에 대해 실현 가능성을 적용함으로써 견고성과 유연성을 보장한다.

* 이 논문에서는 **MILP 문제를 가정**하여 Robust counterpart optimization formulations가 제시된다.

![](https://media.vlpt.us/images/aioptlab/post/022ca964-0b73-419f-8069-fa1344b0995b/image.png)

* 위 가장 기본적인 MILP 공식에서 **데이터의 불확실성이 The left-hand-side의 요소에만 영향을 미친다고 가정**한다.

(1) 목적함수는 제약조건으로 변환될 수 있다.
(2) The right-hand-side 값이 불확실한 경우 고정값이 1인 Xn+1을 도입해 원래 제약조건을 아래와 같이 변경할 수 있다.

![](https://media.vlpt.us/images/aioptlab/post/4a00bec2-772d-429a-b959-af33c46ae442/image.png)

(3) X의 계수가 불확실한 파라미터라고 가정하면 아래와 같이 전개된다.

![](https://media.vlpt.us/images/aioptlab/post/9a23e60a-b32f-44bc-8500-a272c2b5c83c/image.png)

![](https://media.vlpt.us/images/aioptlab/post/a08507e2-f9b0-4a05-8298-5747bd67695b/image.png)


## 1. Soyster's Formulation
* 이 접근방식은 모든 잠재적 실현 가능성을 보장하기 때문에 가장 보수적인 접근 방식이다.

![](https://media.vlpt.us/images/aioptlab/post/59ee4407-f2c8-4f11-a9ac-4eae45a961eb/image.png)

* 위와 같은 Formulation에서의 제약식의 목적은 **불확실한 계수의 모든 가능한 값에 대해 솔루션이 가능한 상태로 유지되도록 보장**하는 것이다.

* 해당 Formulation을 사용하는 이유는 **불확실성으로부터 최대한의 보호(매우 보수적)** 를 받기 위함이다. 하지만 데이터의 모든 실현 가능성이 고려되기 때문에 솔루션이 지나치게 비관적일 수 있으며 문제가 실현 불가능할 가능성이 높아진다.

  -> 예를들어, 모든 처리시간과 모든 수요가 Maximum을 취하는 경우, Plant가 고정된 시간 범위 내에서 수요를 충족할 수 없기 때문에, 실행 불가능한 Schedule이 발생할 수 있다.


## 2. Ben-Tal and Nemirovski's Formulation
* Ben-Tal and Nemirovsiki의 Formulation은 아래와 같다.

![](https://media.vlpt.us/images/aioptlab/post/1e86d5bf-66d0-4905-b13b-efc2da782d12/image.png)

* 불확실한 계수의 값은 Random perturbations를 통해 구한다고 가정한다.

![](https://media.vlpt.us/images/aioptlab/post/c16628e6-3102-40ca-83f9-448f2b99d2c5/image.png)

* Ben-Tal and Nemirovsiki[2] 논문을 보면 **Random perturbations은 [-1, 1]에서 대칭적으로 분포하는 독립적인 Random Variables**임을 알 수 있다.
* 그리고 아래 식과 같이 이 **Robust formulation은 l번째 제약이 위반될 확률이 최대 Kl**인 것을 보증한다.

![](https://media.vlpt.us/images/aioptlab/post/d96cd7e0-9fb7-48fd-b20a-f0ebcb954ccf/image.png)

* 이 Robust optimization formulation은 불확실한 선형 계수의 LP(Linear Programming) 문제를 풀기 위해 처음 도입되었다. 이후 [6]에서 MILP 문제로 확장되었다.
* 이러한 Robust formulation은 제약 위반 확률을 통해 솔루션을 내는데 **어느정도의 유연성을 가지고 있다.** 하지만 비선형 최적화 공식에 해당되어 비선형 최적화의 Complication이 생김.

## 3. Bertsimas and Sim's Formulation
* Ben-Tal and Nemirovski's Formulation에서 발생되는 비선형 최적화의 Complication을 피하기 위해 Bertsimas and Sim은 **Budgets으로 설정된 불확실성을 사용해 계수 불확성을 갖는 Robust linear programming을 고려**한다.
* Budget parameter는 제약 내의 불확실한 계수 parameter의 수와 0 사이의 값을 취하며 반드시 정수는 아니다. 
* Budget parameter를 사용함으로써 모든 불확실성을 갖는 계수가 Worst-case를 얻을 가능성이 낮다.

![](https://media.vlpt.us/images/aioptlab/post/257c0e77-225e-421f-bfd0-e72b73602af5/image.png)

![](https://media.vlpt.us/images/aioptlab/post/f0091a46-bd28-4dc4-9c13-c1182a60f044/image.png)

* 위 식은 최대 Budget parameter까지 불확실한 파라미터가 동시에 Worst-case를 얻을 수 있다는 요건을 나타낸다. Budget parameter가 정수로 나타났을 때 명확히 알 수 있다.

* 따라서, Robust formulation은 아래와 같다.

![](https://media.vlpt.us/images/aioptlab/post/76190087-e63b-4ee0-893c-782181595447/image.png)

* (P4)를 단일 최적화 문제로 변환하려면 (10)번 식이 되고, 이 식은 (P5)의 목적함수와 같다.

![](https://media.vlpt.us/images/aioptlab/post/27cb9aeb-2407-41a6-b442-e57e98bacf37/image.png)

![](https://media.vlpt.us/images/aioptlab/post/38af1059-9ebc-481b-9356-8dd1ef156b2f/image.png)

* (P5)를 Dual problem으로 표현하면 (P6)와 같다.

![](https://media.vlpt.us/images/aioptlab/post/0216423d-5e1a-4c42-bc3b-ae21b4a4fedd/image.png)

* 따라서, (P4)의 Inner optimization problem을 (P6)로 대체하면 Robust formulation은 (P7)과 같은 Equivalent formulation으로 변환된다.

![](https://media.vlpt.us/images/aioptlab/post/437cb067-716f-42b5-83d8-2202d777320e/image.png)

* 이 모델에서는 Budget parameter로 Worst-case를 동시에 취할 수 있는 계수의 수를 제한하기 때문에 Linear formulation이 유지된다.

*  그리고 Bertsimas and Sim은 Robust counterpart formulation에 대해, 제약 조건 위반에 대한 확률 경계를 계산했다. 

![](https://media.vlpt.us/images/aioptlab/post/a08507e2-f9b0-4a05-8298-5747bd67695b/image.png)

* 위와 같은 조건을 만족할 때, i번째 제약 조건을 위반할 확률은 아래와 같은 제약 조건을 만족한다.

![](https://media.vlpt.us/images/aioptlab/post/e4df5780-8676-4c07-8be5-cba237f3a4a0/image.png)

* 다수의 불확실한 매개변수를 가진 실제 문제에 이 Robust counterpart formulation을 적용할 때, 우리는 **Feasibility test를 통해 어떤 매개변수가 Worst-case를 취할 수 있는지 식별**할 수 있으며, **다양한 제약조건에 적절한 Budget parameter를 할당하는 지침을 제공**할 수 있다.

## 4. Comparison of Different Formulations
* n개의 변수와 m개의 제약조건, 총 k개의 불확실한 Parameter를 갖는 MILP 문제의 경우, 모든 제약조건의 숫자 j가 불확실성의 대상이고 모든 의사결정 변수의 숫자 q가 불확실성의 대상인 경우, 우리는 3개의 다른 Robust formulation에 대해 변수의 수, 필요한 제약의 수, Formulation 유형 및 얻는 정보의 관점에서 비교한다.

**(1) Soyster's formulation**
* n + q개의 변수와 m + 2q개의 제약조건을 가지며, Linear formulation이지만 솔루션의 보수성을 제어할 수 없다.

* Soyster의 worst-case formation은 **변수와 제약이 최소인 가장 단순한 공식**이지만 **솔루션 보수성을 조정할 수 없기 때문에 생성된 솔루션이 지나치게 비관적일 수 있다.**

**(2) Ben-Tal and Nemirovski's Formulation**
n + 2k개의 변수와 m + 2k개의 제약조건을 갖는 2차 원뿔 문제(비선형)이며, 제약 조건 위반 확률에 대한 파라미터를 통해 보수성의 정도를 제어할 수 있다. 

* **솔루션 보수성에 대한 제어가 가능**하지만, 결과적으로 비선형 공식을 초래하여 **혼합 정수 비선형 프로그래밍(MINLP) 문제를 해결하는 데 있어 계산이 복잡**해진다.

**(3) Bertsimas and Sim's Formulation**
* n + j + k + q개의 변수와 m + k + 2q개의 제약조건을 갖는 선형 최적화 문제이며, Budget parameter를 통해 솔루션의 보수성의 정도를 제어한다.

* 솔루션 보수성을 조정할 수 있는 **유연성을 가진 선형 공식**으로, **문제 크기가 크게 증가하지 않는다.** 모든 Robust counterpart formulation은 Original deterministic formulation과 동일한 수의 이항 변수를 가집니다.

# 3. Robust Scheduling
* 일반적인 공정 스케줄링 문제에 대해서는 Ierapetritou와 Floudas[7]가 제안한 (P8)과 같은 Deterministic formulation이 사용된다.

![](https://media.vlpt.us/images/aioptlab/post/5f1c2912-4f58-4a0b-9673-a2f6f873998b/image.png)

![](https://media.vlpt.us/images/aioptlab/post/20377dec-f829-494b-ad13-41fe062f7fdc/image.png)

(12) : n시점에서 각 단위의 작업을 하나만 수행 가능하다.

(13) : 물질적 균형

(14), (15) : 생산 단위의 저장 및 용량 제한

(16) : Demand 제약

(17)~(24) : 생산 단위에서 Task 지속시간 및 시퀀스 요건에 따른 시간제한

## 1. Price Uncertainty
* Bertsimas and Sim's의 Formulation을 적용하기 위해 아래와 같이 Formulation을 수정한다.

![](https://media.vlpt.us/images/aioptlab/post/331b1656-aa58-413d-af18-8cdf0e1b78b9/image.png)

* 해당 식에서 불확실한 Price의 수(k)는 불확실한 매개변수의 수이기 때문에 Budget parameter의 범위는 [0, k]이다.

![](https://media.vlpt.us/images/aioptlab/post/e29bac73-dbf0-4b2f-a951-df113cf196ad/image.png)

* 위처럼 견고한 제약식은 (P7)의 식과 동일한 형태로 바뀌게되며, d >= 0이고, (STI - STF) >= 0이므로 보조변수는 필요없게 된다.

## 2. Processing Time Uncertainty
* Processing time에 대한 제약조건은 다음과 같다.

![](https://media.vlpt.us/images/aioptlab/post/bcea5fbd-2fee-4c44-a954-b160ff787d76/image.png)

![](https://media.vlpt.us/images/aioptlab/post/9525b2fa-5b32-45bb-a823-59b346dd9f8a/image.png)

* (30)번 식의 불확실한 변수인 θ˜i,j를 Budget parameter를 사용해서 식을 변경하면 (31),(32),(33)의 제약이 생성되는데 이는 (P7)의 제약조건이 된다.

![](https://media.vlpt.us/images/aioptlab/post/05af73d7-cc88-4d2d-887b-e0d485c29735/image.png)

* 따라서 총 변수의 수는 불확실한 제약의 수와 불확실한 파라미터의 수를 더한 값이 된다.

## 3. Demand Uncertainty
* 수요의 불확실성에 대한 제약은 다음과 같다.

![](https://media.vlpt.us/images/aioptlab/post/6e9c815c-f40e-4d73-ad1a-83236a0f11a7/image.png)

* (35)번 식으로 변경 후 Budget parameter를 사용하여 (36),(37),(38)의 제약을 생성한다.

![](https://media.vlpt.us/images/aioptlab/post/b1897597-2541-417c-b0b4-de82b643b5c3/image.png)

![](https://media.vlpt.us/images/aioptlab/post/adf39ec5-e600-40f0-a489-647ad21b64f2/image.png)

# 4. Examples
* 모든 Examples는 Windows XP에서 CPLEX 10.1 솔버를 사용해 문제를 풀었다.

## Example 1

![](https://media.vlpt.us/images/aioptlab/post/965530a0-89a5-4d53-8bb2-658f2337bfa7/image.png)

이 예는 Ierapetritou, Floudas[4]에서 가져온 것으로, **3개의 Feed를 사용하여 2개의 제품을 생산**한다. 이 예를 통해 **다른 유형의 불확실성을 고려**해 이전에 언급된 3가지 Robust counterpart optimization 기법을 비교한다.

### 1. Price Uncertainty
* 원자재 원가와 제품 가격에 대한 불확실성을 전재로 푸는 문제이다. 스케줄링 시간은 8시간이며 모든 원료의 원가는 5, 제품 1과 2의 원가는 10, 15이다. 모든 가격에 대해 5%의 변동 수준이 가정된다.

![](https://media.vlpt.us/images/aioptlab/post/82430a79-8a99-4ef7-bde3-8f57567ecba5/image.png)

* Soyster 공식은 Worst-case(제품 1,2의 최소 가격은 9.5, 14.25 / 원자재 비용의 최대값은 5.25, 5.25, 5.25)에 대해 푼 결과로 959.56이다.
* Ben-Tal 공식은 제약을 위반할 확률을 최대 28.4%와 10%의 두가지 경우로 문제를 풀었다. 
* Bertsimas-Sim 공식은 0~5사이의 Budget parameter를 사용해 풀었다.

* **결과적으로, 제약 위반 최대 확률이 같을 때, Bertimas-Sim의 공식이 Ben-Tal보다 더 높은 이익을 창출하며, 이는 Ben-Tal 공식이 더 보수적임을 의미한다.**

### 2. Processing Time Uncertainty
* 모든 처리 시간이 불확실한 파라미터이며 15%의 변동성을 갖는다고 가정한다. 원재료의 원가는 0이며, 제품 1과 2의 가격은 10으로 설정되어있다.

![](https://media.vlpt.us/images/aioptlab/post/064f7513-505b-410e-9de1-0f054abf08ca/image.png)

* Soyster 공식과 Ben-Tal의 공식은 Deterministic formulation과 비교했을 때, 변수와 제약조건이 동일하다. 이는 불확실한 매개 변수는 이항 변수의 계수이고, 보조 변수가 추가되지 않기 때문이다.
* Bertimas-Sim의 공식은 더 많은 제약과 변수를 가지고 있지만, 공식의 선형성이 있어 효율적이다.

### 3. Demand Uncertainty
* 이 문제는 제품 1과 2의 수요가 모두 50%의 변동수준을 가지며, 둘의 범위는 [25, 75]이고, Nominal value는 50이다.
* 신뢰도 수준이 작을수록 문제를 풀 수 없기 때문에 Ben-Tal 공식의 신뢰도 수준은 75%로 설정했다.

![](https://media.vlpt.us/images/aioptlab/post/baf1f3ff-e786-40cd-89d6-f117d178d0f7/image.png)

* Soyster의 공식은 해당 문제에서 infeasible한 결과를 얻게 되는데 이것은 Worst-case에서는 infeasible하다는 것을 의미한다.
* Bertsimas-Sim의 Robust formulation은 Worst-case의 실현 가능성이 존재할 때, 다른 Budget parameter에서도 실현가능하다. 하지만 Soyster 공식과 마찬가지로 Bertsimas-Sim의 공식에서도 Budget parameter가 가장 큰 값으로 설정되었을 때, infeasible한 결과를 얻는다.

---
#### * 세가지 비교에 대한 요약
* 제약 조건과 변수의 수의 증가가 불확실한 매개변수의 수와 동일한 규모이기 때문에 모든 Robust formulation의 크기가 크게 증가하지 않는다.
* Scheduling formulation에서 대부분의 의사결정 변수는 양의 변수이거나 이진수이기 때문에, 보조 변수의 수와 보조 변수의 제약 조건을 줄일 수 있다.
* Soyster의 Worst-case formulation은 가장 보수적인 제형이며 보수성의 정도를 조정할 수 있는 유연성이 없다. 
* Ben-Tal의 공식은 제약 위반 확률로 보수성의 정도를 조정할 수 있다.
* Bertimas-Sim의 공식은 처리 시간과 수요 불확실성을 대처할 때 상대적으로 더 많은 제약과 연속 변수를 포함하지만, 보수성의 정도를 제어하는데 더 유연하고, 비선형의 문제를 피할 수 있으므로 효율성이 크게 향상된다.
---

### 4. Systematically Considering All Uncertainties
* 모든 가공시간 15%, 제품1과 제품2의 수요 50%, 제품1과 제품2의 가격 5% 등 모든 불확실한 파라미터를 동시에 고려한다.
* 모든 불확실성을 고려하여 문제를 해결하기 위해 몇 가지 다른 Budget parameter 조합이 사용해 **<Table 4>** 에서 Profit objective와 Budget parameter의 관계를 나타내고 있다.

![](https://media.vlpt.us/images/aioptlab/post/f3965f79-e14a-4f06-b04b-558b88004e37/image.png)

* Budget parameter가 높을수록 실현 가능성은 커지지만 이익은 작아지는 것을 볼 수 있다.
* Budget parameter의 조합이 (0.5, 0.3, 0.3)이면 가격은 61.3%, 수요는 67.5%, 기간은 67.5% 확률로 제약 조건을 위반할 수 있다. 이에 따라 이익은 감소한다.

![](https://media.vlpt.us/images/aioptlab/post/c472421a-207f-4dba-912f-0d59226c0ef1/image.png)

* 제품 2가 제품 1보다 가치가 높기 때문에 **<Figure 2>** 에서는 제품 1을 52, 제품 2를 87.5만큼 생산하고자 한다.

* 하지만 제품 1, 2의 수요가 [25, 75]의 범위에 있기 때문에 불확실한 범위를 포함하는 Feasible한 스케줄은 **<Figure 3>** 과 같으며, 제품 1과 2를 각각 58, 70.3만큼 생산한다.

![](https://media.vlpt.us/images/aioptlab/post/db5db40c-ffd3-4e7f-8780-401453c66cb9/image.png)

* **<Figure 2>, <Figure 3>** 을 비교해보면 Separation 단계에서 **<Figure 3>** 에서의 처리되는 재료의 양이 적기 때문에 공정을 고려할 때 주어진 시간 범위 내에 작업이 완료된다는 것을 알 수 있다.

## Example 2

![](https://media.vlpt.us/images/aioptlab/post/752ac7c0-0437-432c-abca-06b2d8440df4/image.png)

이 예는 Wu, Ierapetritou[5]에서 가져온 것으로 **3개의 Feed로 8개의 Task를 통해 4개의 제품을 생산**한다. 그리고 **중간에는 9개의 Intermediates(중간체)가 존재**한다. **전체 공정에서 총 6개의 다른 Units도 필요**하다.

* 스케줄링 시간은 18시간이며 모든 처리 시간의 Nominal value는 1, 제품 1,2,3,4의 가격은 18, 19, 20, 21이며, 수요는 6000, 8000, 2000, 8000이다.
* 모든 처리시간은 20%의 변동 수준을 가지며, 제품 가격은 10%의 변동 수준을 가진다.

![](https://media.vlpt.us/images/aioptlab/post/780a11a3-759a-4642-9ff0-18b5755e0a27/image.png)

* Nominal value를 활용한 스케줄링의 결과는 **<Figure 5>** 와 같다. 제품 1, 2, 3, 4의 생산량은 7522, 16418, 2835, 11180이다.

![](https://media.vlpt.us/images/aioptlab/post/b0bf47e4-e861-47e8-9c24-4da546d98cbe/image.png)

* **<Figure 6>** 에서 제품 1, 2, 3, 4의 생산량은 7200, 15096, 2522, 11256이며, **<Figure 7>** 에서 제품 1, 2, 3, 4의 생산량은 7150, 15400, 2459, 1023이다.

![](https://media.vlpt.us/images/aioptlab/post/46f53958-e87f-41b6-a1d9-dd1ac05c8edc/image.png)

![](https://media.vlpt.us/images/aioptlab/post/5ffb371f-dc38-4b47-a11b-0af18139a51c/image.png)

* Budget parameter 값이 증가함에 따라 **Robust schedule의 생산은 불확실한 처리시간 하에서 지속시간 요건을 충족시키기 위해 생산량을 감소시키는 경향이 있어 총 이익이 감소**되는 것을 알 수 있다.
* Budget parameter가 증가하여 보수적인 솔루션이 될 경우 계산시간이 길어지는데 이는 실현 가능성 공간이 작아지기 때문이다.

# 5. Summary
 프로세스 스케줄링 문제에서 발생하는 다양한 불확실성에 대처하는 Robust preventive schedule을 생성하기 위한 연구는 주로 **시나리오 기반의 확률적 스케줄링**으로 이뤄졌었다. 하지만 이 방법은 **불확실성을 갖는 parameter가 증가할 때마다 문제의 크기가 기하급수적으로 증가한다는 단점이 있다.**
 이러한 단점을 극복하기 위해 **Robust counterpart optimization**이 등장했다. 이 논문에서는 3가지의 Robust counterpart optimization을 연구하고 성능을 비교했다. 그 결과 **Bertsimas, Sim[3]에 의해 제안된 Formulation이 불확실성을 갖는 스케줄링 문제에 적합하며, 문제의 크기를 실질적으로 증가시키지 않고, 선형성을 유지하며, 모든 제약 조건에 대한 conservatism의 정도를 제어하고 Budget paraneter를 사용하여 Robust optimization 문제에 대한 실현 가능성을 보장한다.**

# Nomenclature
![](https://media.vlpt.us/images/aioptlab/post/cd594dd4-a61f-4fec-af58-f1fda3b8ab49/image.png)

![](https://media.vlpt.us/images/aioptlab/post/a22f62f7-a0a2-4c86-b3c9-bc06035a2c2c/image.png)

# Reference
[1] Soyster, A. L. Convex programming with set-inclusive constraints and applications to inexact linear programming. Oper. Res. 1973, 21, 1154.
[2] Ben-Tal, A.; Nemirovski, A. Robust solutions of Linear Programming problems contaminated with uncertain data. Math. Prog. 2000, 88, 411.
[3] Bertsimas, D.; Sim, M. Robust Discrete optimization and Network Flows. Math. Prog. 2003, 98, 49.
[4]  Ierapetritou, M. G.; Floudas, C. A. Effective continuous-time formulation for short-term scheduling. 1. Multipurpose batch processes. Ind. Eng. Chem. Res. 1998, 37, 4341.
[5]  Wu, D.; Ierapetritou, M. G. Decomposition approaches for the efficient solution of short-term scheduling problems. Comput. Chem. Eng. 2003, 27, 1261.

[6] Lin, X.; Janak, S. L.; Floudas, C. A. A new robust optimization approach for scheduling under uncertainty: I. bounded uncertainty. Comput. Chem. Eng. 2004, 28, 1069.

[7]  Ierapetritou, M. G.; Floudas, C. A. Effective continuous-time formulation for short-term scheduling. 1. Multipurpose batch processes. Ind. Eng. Chem. Res. 1998, 37, 4341.
