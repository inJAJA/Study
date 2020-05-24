"""
: 분석가에게 파라미터에 대한 사전분포가 주어지고, 
관측된 데이터와 베이즈 정리를 사용하여 사후분포는 갱신할 수 있다.

알려지지 않은 파라미더타 확률이라고 하면 보통 모든 확률 값이 0과 1사이에서 정의되는
'베타 분포'를 사전 분포로 사용한다.
"""
import math

def B(alpha: float, beta: float) -> float:
    """모든 확률값의 합이 1이 되도록 해주는 정규화 값"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1 :                    # [0, 1] 구간 밖에서는 밀도가 없다.
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)

"""
alpha = beta = 1 : 균등분포( 중심이 0.5이고 굉장히 퍼진 )
alpha > beta = 1 : 대부분의 밀도는 1 근처에
alpha < beta = 1 : 대부분의 밀도는 0 근처에

"""