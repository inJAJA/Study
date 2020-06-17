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

    recipe_title = [] # 레시피 제목
    recipe_source = [] # 레시피 재료
    recipe_step = [] #레시피 순서

    
    res = soup.find('div', {'class':'top'})
    title = res.find('h1')
    recipe_title = title.find_all('strong')

    res = soup.find('ul', {'class': 'lst_ingrd'})
 
    for tmp in res.find_all('span'):
        recipe_source.append(tmp.get_text().replace('\n','').replace(' ',''))
    print(recipe_source)
 
    # 요리 순서 찾는 for문
    res = soup.find('ol', {'class':'lst_step'})
    i = 0
    for n in res.find_all('p'):
        i += 1
        recipe_step.append('#' + str(i)+' '+ n.get_text())

    recipe_all = {'recipe_title': recipe_title, 'recipe_source':[recipe_source], 'recipe_step':[recipe_step]}
    return(recipe_all)
    
recipe = PageCrawler('5944')
recipe_list = pd.DataFrame(recipe)
print(recipe_list.info())

recipe.to_csv('recipe.csv')
