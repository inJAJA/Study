import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
X = dataset.data
Y = dataset.target

print(X.shape)
print(Y.shape)

pca = PCA(n_components= 5)
x2 = pca.fit_transform((X))
pca_evr = pca.explained_variance_ratio_ # PCA가 설명하는 분산의 비율을 확인
print(pca_evr)                          # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856]
print(sum(pca_evr))                     # 0.8340156689459766    
                                        # 압축률 / 약 0.17의 손실이 생김 = 필요없는 값(0)이 손실된 것임으로 큰 상관 X