from flask import Flask

app = Flask(__name__)

@app.route('/') # @app.route('/')경로에 다음 함수가 나오도록 함
def hello333():
    return "<h1>hello youngsun world</h1>"            # h1 : 글씨 크기 크게

@app.route('/bit')
def hello334():
    return "<h1>hello bit computer world</h1>"

@app.route('/gema')
def hello335():
    return "<h1>hello GEMA computer world</h1>"

@app.route('/bit/bitcamp')
def hello336():
    return "<h1>hello bitcamp world</h1>"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8888, debug=True)  # 200 : 정상 구동
                # IP 


