import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
X = dataset.data
Y = dataset.target

print(X.shape)
print(Y.shape)

# pca = PCA(n_components= 5)
# x2 = pca.fit_transform((X))
# pca_evr = pca.explained_variance_ratio_ 
# print(pca_evr)                          
# print(sum(pca_evr))                       

pca = PCA()
pca.fit(X) 
cumsum = np.cumsum(pca.explained_variance_ratio_)   # np.cumsum() : 누적 계산
print(cumsum)                                       # pca를 썼을 때 몇개로 줄일지 판단 가능 / n개로 줄인 만큼의 pca값들
                                                    # [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 
                                                    #  0.89428759 0.94794364 0.99131196 0.99914395 1.        ]

aaa = np.argmax(cumsum >= 0.94) + 1                 # 0.94 이상의 압축률을 가진 n_components를 반환
print(cumsum >= 0.94)                               # [False False False False False False  True  True  True  True]
print(aaa)                                          # 7
