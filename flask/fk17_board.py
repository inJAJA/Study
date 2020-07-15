from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# 데이터 베이스
conn = sqlite3.connect('./data/wanggun.db')                 # DB wanggun에 접속
cursor = conn.cursor()
cursor.execute('SELECT * FROM general')                     # general에 있는 모든 데이터 가져오기
print(cursor.fetchall())

@app.route('/')
def run():
    conn = sqlite3.connect('./data/wanggun.db')             # app에서 다시 접속
    c = conn.cursor()
    c.execute('SELECT * FROM general;')
    rows = c.fetchall()
    return render_template('board_index.html', rows = rows)

@app.route('/modi')                                         # 이름을 눌렀을 때 전달해주는 구간
def modi():
    id = request.args.get('id')                             # id를 요청하여 넣어준다
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id = '+str(id))   # WHERE : 선택 
    rows = c.fetchall()
    return render_template('board_modi.html', rows = rows)

@app.route("/addrec", methods = ["POST", "GET"])                # 실질적 수정 부분
def addrec():
    if request.method == 'POST':
        print(request.form['war'])
        print(request.form['id'])
        try:
            war = request.form['war']
            id = request.form['id']
            with sqlite3.connect("./data/wanggun.db") as con:
                cur = con.cursor()
                cur.execute("UPDATE general SET war="+str(war)+" WHERE id="+str(id))    # war의 컬럼을 수정
                con.commit()                                                            # 수정 후 항상 커밋
                msg="정상적으로 입력되었습니다"

        except:                                                 # try 부분에서 error가 뜨면 예외 처리 해준다
            con.rollback()                                      # error가 뜨면 rollback한다.
            msg="입력과정에서 에러가 발생했습니다."

        finally:
            return render_template("board_result.html",msg = msg)
            con.close()

app.run(host = '127.0.0.1', port =5011, debug=False)
'''입력 과정에서 에러 뜸'''
