import  pandas as pd

data = {'city': ["Nagano","Sydney","Salt Lake city","Athens","Torino"
                ,"Beijing","Vancouver","London","Sorchi","Rio de Janeiro"],
        'year':["1998",'2000','2002','2004','2006',
                '2008','2010','2012','2014','2016'],
        'season':['winter','summer','winter''summer','winter',
                  'summer','winter','summer','winter','summer']}

df = pd.DataFrame(data)

df.to_csv("csv1.csv")

# 문제
data = {'OS': ['Machintosh', 'Windows','Linux'],
        'relese': [1984, 1985, 1991],
        'country': ['US','US','']}

df = pd.DataFrame(data)
df.to_csv("OSlist.csv")