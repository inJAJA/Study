"""
#  선 그래프( line chart ) : plt.plot()
: 어떤 경향을 보여 줄 떄 유용
"""
variance     = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error  = [x + y for x, y in zip(variance, bias_squared)]
xs = [ i for i, _ in enumerate(variance)]                # enumerate는 p037참고 


"""
# 한 차트에 여러 개의 번을 그리기 위해 
# plt.plot을 여러 번 호출 할 수 있다.
"""
from matplotlib import pyplot as plt
plt.plot(xs, variance,      'g-', label='variance')      # 실성
plt.plot(xs, bias_squared, 'r-.', label='bias^2')        # 일점 쇄선
plt.plot(xs, total_error,   'b:', label='total error')   # 점선
"""
# 선 스타일

 문자열	의미
 --------------------
' -  '	solid line style
' -- '	dashed line style
' -. '	dash-dot line style
' :  '  dotted line style

"""
# 각 선에 레이블을 미리 달아 놨기 때문에
# 범례(legend)를 쉽게 그릴 수 있다.
plt.legend(loc =9)
plt.xlabel("model complexity")                           # x축 레이블
plt.xticks([])                                           # x축 레이블 0
plt.title("the Bias-Variance Tradeoff")                  # y축 레이블

plt.show()
