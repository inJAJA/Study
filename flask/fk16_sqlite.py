# 간략화 된 놈
import sqlite3 

conn = sqlite3.connect('test.db') # 만든적 없으면 자동 생성됌

cursor = conn.cursor()
                                # 존재하지 않으면 생성
cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT,
                FoodName TEXT, Company TEXT, Price INTEGER)""")                        # table생성

sql ="DELETE FROM supermarket"              # 계속 실행할 때마다 중복 저장됨 -> 싹 다 지우고 다시 생성
cursor.execute(sql)

# 데이터 넣자
sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (1, '과일','자몽','마트', 1500))

sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (2, '음료수','망고주스','편의점', 1000))

sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (33, '고기','소고기','하나로마트', 10000))

sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (4, '박카스','약','약국', 500))

sql = 'SELECT * FROM supermarket'
# sql = 'SELECT Itemno, Category, FoodName, Company, Price FROM supermarket'
cursor.execute(sql)

rows = cursor.fetchall()

for row in rows:
    print(str(row[0])+' '+str(row[1])+' '+str(row[2])+' '+
         str(row[3]) + ' '+ str(row[4]))

conn.commit()                                                           # DB browser sql에 보낸다
conn.close()

'''
# DB Browser for SQLite
1. 새 데이터 베이스
2. 데이터 보기
'''

