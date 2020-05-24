""" 
문자열(string)은 작은 따옴표(') 또는 큰 따옴표(")로 묶어 나타냄
앞뒤로 동일한 기호 사용해야 한다.
"""

single_quoted_string = 'data science'
single_quoted_string = "data science"

tab_string = "\t"                     # 탭(tab)을 의미하는 문자열
print(len(tab_string))                # 1


not_tab_string = r"\t"                # 문자 '\'와 't'를 나타내는 문자열
print(len(not_tab_string))            # 2


"""세 개의 따옴표를 사용하면 하나의 문지열을 여러줄 로 나타낼 수 있다."""
multi_line_string = """This is the first line.
                       and this is the second line
                       and this is the third line """


first_name = "Joel"
last_name = "Grus"

full_name1 = first_name + " " + last_name             # 문자열 합치기
print(full_name1)                                     # "Joel Grus" 출력

full_name2 = "{0} {1}".format(first_name, last_name)  # .format을 통한 문자열 합치기 
print(full_name1)                                     # "Joel Grus" 출력

full_name3 = f"{first_name} {last_name}"              # f-string( f ) 사용하여 합치기
print(full_name3)                                     # "Joel Grus" 출력

