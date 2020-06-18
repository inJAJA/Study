import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# image로 인식한 재료
ingredients = np.load('./mini_project/graph/ingredient.npy')
ingred = []
for source in ingredients:
    if source == 'carrot':
        source = '당근'
    elif source == 'chicken':
        source = '닭'
    elif source == 'egg':
        source = '계란 달걀'
    elif source == 'fish':
        source = '고등어 삼치 꽁치'
    elif source == 'flour':
        source = '밀가루'
    elif source == 'mashroom':
        source = '버섯'
    elif source == 'meat':
        source = '고기 소고기 돼지고기 '
    elif source == 'onion':
        source = '양파'
    elif source == 'paprika':
        source = '파프리카'
    else:
        source = '감자'
    ingred.append(source)

print(ingred)


# recipe_data
ca = pd.read_csv('./mini_project/recipe/recipe_carrot.csv', encoding= "utf-8", engine ='python')
ch = pd.read_csv('./mini_project/recipe/recipe_chicken.csv',  encoding= "utf-8", engine ='python')
egg = pd.read_csv('./mini_project/recipe/recipe_egg.csv',  encoding= "utf-8", engine ='python')
fs = pd.read_csv('./mini_project/recipe/recipe_fish7.csv',  encoding= "utf-8", engine ='python')
fl = pd.read_csv('./mini_project/recipe/recipe_flour8.csv',  encoding= "utf-8", engine ='python')
ms = pd.read_csv('./mini_project/recipe/recipe_mashroom.csv',  encoding= "utf-8", engine ='python')
mt = pd.read_csv('./mini_project/recipe/recipe_meat.csv', encoding= "utf-8", engine ='python')
on = pd.read_csv('./mini_project/recipe/recipe_onion.csv',  encoding= "utf-8", engine ='python')
pa = pd.read_csv('./mini_project/recipe/recipe_paprika.csv',  encoding= "utf-8", engine ='python')
po = pd.read_csv('./mini_project/recipe/recipe_potato.csv',  encoding= "utf-8", engine ='python')

recipe = pd.concat([ca, ch, egg, fs, fl, ms, mt, on, pa, po]).dropna()
recipe['recipe_source'] = recipe['recipe_source'].apply(literal_eval)

recipe['recipe_source'] = recipe['recipe_source'].apply(lambda x: " ".join(x))

# 단어 토큰화
vector = CountVectorizer(ngram_range=(1, 1))                # 단어 묶음을 1개부터 1개까지 설정
vector.fit(ingred)                                  
c_vector_recipe = vector.transform(recipe['recipe_source']) # 변환
c_vector_ingred = vector.transform(ingred)

# 코사인 유사도를 구한 벡터를 미리 저장
ingred_c_sim = cosine_similarity(c_vector_ingred, c_vector_recipe).argsort()[:, ::-1] # 오름차순 정렬
print(ingred_c_sim.shape)

sim_index = ingred_c_sim[:3]
print(sim_index)

recipe_recommend = [i[0]for i in sim_index]

print(recipe.iloc[recipe_recommend])

