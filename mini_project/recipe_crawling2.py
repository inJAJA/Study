import numpy as np
import urllib
from bs4 import BeautifulSoup
import csv
import pandas as pd

def PageCrawler(recipeUrl):
    url = 'https://haemukja.com/recipes/' + recipeUrl

    req = urllib.request.Request(url)
    sourcecode = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(sourcecode, "html.parser")

    recipe_title = []  # 레시피 제목
    recipe_source = [] # 레시피 재료
    recipe_step = []   #레시피 순서

    
    res = soup.find('div', {'class':'top'})
    title = res.find('h1')
    recipe_title = title.find('strong')
    recipe_title = recipe_title.get_text()

    res = soup.find('ul', {'class': 'lst_ingrd'})
 
    for tmp in res.find_all('span'):
        recipe_source.append(tmp.get_text().replace('\n','').replace(' ',''))
 
    # 요리 순서 찾는 for문
    res = soup.find('ol', {'class':'lst_step'})
    i = 0
    for n in res.find_all('p'):
        i += 1
        recipe_step.append('#' + str(i)+' '+ n.get_text())

    # 레시피 생성
    recipe_all = {'recipe_title': recipe_title, 'recipe_source':[recipe_source], 'recipe_step':[recipe_step]}
    return(recipe_all)

# 빈 데이터 프레임 만들기
recipe_tol = pd.DataFrame(index = range(0, 1), 
                          columns = ['recipe_title', 'recipe_source', 'recipe_step'])

# 레시피 생성
for i in [48,5810,405,1754,5167,2169,3344,4336,3348,5600] :  # 페이지 주소
    print(i)
    recipe = PageCrawler(str(i))
    recipe_list = pd.DataFrame(recipe)
    recipe_tol = pd.concat([recipe_tol, recipe_list])


recipe_tol.to_csv('./mini_project/recipe/recipe_meat.csv', index = False)
