import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ingredients = np.load('./data/ingredient.npy')

recipe['ingred'] = recipe.apply(lambda x : ''.join(x)) # 띄어쓰기로 이루어진 str로 변경

recipe.ingred.head(2)

count_vector = CountVectorizer(ngram_range=(1, 3)) # 객체 생성
c_vector_ingred = count_vector.fit(recipe['ingred']) # 변환
print(c_vector_ingred.shape)

# 코사인 유사도를 구한 벡터를 미리 저장
ingred_c_sim = cosine_similarity(c_vector_ingred, c_vector_ingred).argsort()[:, ::-1]
print(ingred_c_sim.shape)

def get_recommend_recipe_list(df, recipe_title, top=3):    # data, data_title
    # 특정 영화와 비슷한 영화를 추천해야 하기 때문에 '특정 레시피'정보를 뽑아낸다.
    target_recipe_index = df[df['title']] == recipe_title.index.ValuesView

    # 코사인 유사도 중 비슷한 코사인 유사도를 가진 정보를 뽑아낸다.
    sim_index = ingred_c_sim[target_recipe_index, :top].reshape(-1)
    
    # data frame으로 만들고 return
    result = df.iloc[sim_index][:10]
    return result

get_recommend_recipe_list(ingredients, recipe_title= '')