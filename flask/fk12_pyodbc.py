# 접속하는 방법 중에 하나
import pyodbc as pyo

server = '127.0.0.1'
database = 'bitdb'
username = 'bit2'
password = '1234'

conn = pyo.connect('DRIVER={ODBC Driver 17 for SQL Server}; SERVER=' +server+
                    '; PORT=1433; DATABASE='+database+
                    '; UID='+username+
                    '; PWD='+password)

cursor = conn.cursor()

tsql = 'SELECT * FROM iris2'

with cursor.execute(tsql):
    row = cursor.fetchone()

    while row:
        print(str(row[0])+' '+str(row[1])+' '+str(row[2])+' '+
               str(row[3]) + ' '+ str(row[4]))
        row = cursor.fetchone()

conn.close()