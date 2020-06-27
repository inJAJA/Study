import sys
print(sys.path)
'''
import 경로
['d:\\Study\\module', 
'C:\\Users\\bitcamp\\anaconda3\\python37.zip', 
'C:\\Users\\bitcamp\\anaconda3\\DLLs', 
'C:\\Users\\bitcamp\\anaconda3\\lib', 
'C:\\Users\\bitcamp\\anaconda3', 
'C:\\Users\\bitcamp\\AppData\\Roaming\\Python\\Python37\\site-packages',
 'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages', 
 'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages\\win32', 
 'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages\\win32\\lib', 
 'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages\\Pythonwin']
'''

from test_import import p62_import
p62_import.sum2()
# 이 import는 아나콘다 폴더에 들어있닷
# 작업그룹 인포트 썸탄다.

from test_import.p62_import import sum2
sum2()
# 작업그룹 인포트 썸탄다.