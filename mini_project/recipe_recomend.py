import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# recipe_data
ca = pd.read_csv('./mini_project/recipe/recipe_carrot.csv', header = 1)
ch = pd.read_csv('./mini_project/recipe/recipe_chicken.csv', header = 1)
egg = pd.read_csv('./mini_project/recipe/recipe_egg.csv', header = 1)
fs = pd.read_csv('./mini_project/recipe/recipe_fish.csv', header = 1)
fl = pd.read_csv('./mini_project/recipe/recipe_flour.csv', header = 1)
ms = pd.read_csv('./mini_project/recipe/recipe_mashroom.csv', header = 1)
mt = pd.read_csv('./mini_project/recipe/recipe_meat.csv', header = 1)
on = pd.read_csv('./mini_project/recipe/recipe_onion.csv', header = 1)
pa = pd.read_csv('./mini_project/recipe/recipe_paprika.csv', header = 1)
po = pd.read_csv('./mini_project/recipe/recipe_potato.csv', header = 1)

recipe = pd.concat([ca, ch, egg, fs, fl, ms, mt, on, pa, po])

recipe['recipe_source'] = recipe['recipe_source'].apply(lambda x : ''.join(x)) # 띄어쓰기로 이루어진 str로 변경

recipe.ingred.head(2)

count_vector = CountVectorizer(ngram_range=(1, 3)) # 객체 생성
c_vector_ingred = count_vector.fit(recipe['recipe_source']) # 변환
print(c_vector_ingred.shape)

# 코사인 유사도를 구한 벡터를 미리 저장
ingred_c_sim = cosine_similarity(c_vector_ingred, c_vector_ingred).argsort()[:, ::-1]
print(ingred_c_sim.shape)

def get_recommend_recipe_list(df, recipe_title, top=3):    # data, data_title
    # 특정 영화와 비슷한 영화를 추천해야 하기 때문에 '특정 레시피'정보를 뽑아낸다.
    target_recipe_index = df[df['recipe_title']] == recipe_title.index.ValuesView

    # 코사인 유사도 중 비슷한 코사인 유사도를 가진 정보를 뽑아낸다.
    sim_index = ingred_c_sim[target_recipe_index, :top].reshape(-1)
    
    # data frame으로 만들고 return
    result = df.iloc[sim_index][:10]
    return result

get_recommend_recipe_list(ingredients, recipe_title= '')