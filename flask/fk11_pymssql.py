# sql에서 data가져오기
import pymssql as ms                                    # pip install pymssql
print('잘 접속 됬지')
'''
# 아이디 접속 설정
: 서버 드럼통 -> 속성:보안 -> sql sever 및 window 인증모드로 설정 
    -> 작업관리자:서비스 -> MSSQL$... 다시시작
'''

conn= ms.connect(server='127.0.0.1', user='bit2',       # mssql에 연결
                password='1234', database='bitdb')

cursor = conn.cursor()

# cursor.execute('SELECT * FROM iris2;')
# cursor.execute('SELECT * FROM iris2;')
cursor.execute('SELECT * FROM sonar;')


row = cursor.fetchone()                                 # .fetchone() : 한줄을 가져 온다

while row :
    # print('첫컬럼 : %s, 둘컬럼 : %s'%(row[0], row[1]))
    print('첫컬럼 : %s, 둘컬럼 : %s 세컬럼 : %s'%(row[0], row[1], row[2]))
    row = cursor.fetchone()

conn.close()                                            # 연결 끊기

'''
# port 설정
: sql 구성 관리자 설정 -> 네트워크 구성 -> sql에 대한 프로토콜 -> TCP/IP 사용 
   -> 속성:IP주소 -> IPAII: TCP 동적 port 삭제 / TCP포트 1433(기본 port) 설정
'''