import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
'''
# image로 인식한 재료
ingredients = np.load('./data/ingredient.npy')
ingred = []
for source in ingredients:
    if source == 'carrot':
        source = '당근'
    elif source == 'chicken':
        source = '닭'
    elif source == 'egg':
        source = '계란'
    elif source == 'fish':
        source = '고등어'
    elif source == 'flour':
        source = '밀가루'
    elif source == 'mashroom':
        source = '버섯'
    elif source == 'meat':
        source = '고기'
    elif source == 'onion':
        source = '양파'
    elif source == 'paprika':
        source = '파프리카'
    else:
        source = '감자'
    ingred.append([source])

print(ingred)
'''
ingred = ['고기','버섯','밀가루','당근']
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

recipe['recipe_source'] = recipe['recipe_source'].apply(lambda x : ''.join(x)) # 띄어쓰기로 이루어진 str로 변경

vector = CountVectorizer(ngram_range=(1, 3)) # 객체 생성
vector.fit(recipe['recipe_source'])
c_vector_recipe = vector.transform(recipe['recipe_source']) # 변환
c_vector_ingred = vector.transform(ingred)

# 코사인 유사도를 구한 벡터를 미리 저장
ingred_c_sim = cosine_similarity(c_vector_ingred, c_vector_recipe)

get_recommend_recipe_list = ingred_c_sim[ingred]
print(ingred_c_sim.shape)
'''
def get_recommend_recipe_list(df, ingred, top=3):    # data, data_title
    # 특정 영화와 비슷한 영화를 추천해야 하기 때문에 '특정 레시피'정보를 뽑아낸다.
    target_recipe_index = ingred

    # 코사인 유사도 중 비슷한 코사인 유사도를 가진 정보를 뽑아낸다.
    sim_index = ingred_c_sim[target_recipe_index, :top].reshape(-1)
    
    # data frame으로 만들고 return
    result = df.iloc[sim_index][:10]
    return result
'''
get_recommend_recipe_list(ingred, recipe_title= '')