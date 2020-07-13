from flask import Flask, Response, make_response
app = Flask(__name__)

@app.route('/')
def response_test():
    custom_response = Response('Custom Response', 200, 
                    {'Program' : 'Flask Web Application'})
    print('[★ ] app.route')
    return make_response(custom_response)

@app.before_first_request
def before_first_request():
    print('[1] 앱이 가동되고 나서 첫번째 HTTP 요청에만 응답합니다.')    # 앱이 실행되면 제일 먼저 돌아간다
    print('    이 서버는 개인 자산이니 건들지 말것')
    print('    곧 자료를 전송합니다')
#  * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
# [1] 앱이 가동되고 나서 첫번째 HTTP 요청에만 응답합니다.               # before_first_request 
#     이 서버는 개인 자산이니 건들지 말것
#     곧 자료를 전송합니다
# [★ ] app.route
# 127.0.0.1 - - [13/Jul/2020 16:25:22] "GET / HTTP/1.1" 200 -    # app.route 

@app.before_request
def before_request():
    print('[2] 매 HTTP 요청이 처리되기 전에 실행됩니다')
#  * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
# [1] 앱이 가동되고 나서 첫번째 HTTP 요청에만 응답합니다.
#     이 서버는 개인 자산이니 건들지 말것
#     곧 자료를 전송합니다
# [2] 매 HTTP 요청이 처리되기 전에 실행됩니다                           # 매번 실행됨
# [★ ] app.route
# 127.0.0.1 - - [13/Jul/2020 16:32:59] "GET / HTTP/1.1" 200 -
# 매 HTTP 요청이 처리되기 전에 실행됩니다                           # 매번 실행됨
# 127.0.0.1 - - [13/Jul/2020 16:33:16] "GET / HTTP/1.1" 200 -

@app.after_request
def after_request(response):
    print('[3] 매 HTTP 요청이 처리되고 나서 실행됩니다')
    return response
# [2] 매 HTTP 요청이 처리되기 전에 실행됩니다
# [★ ] app.route
# [3] 매 HTTP 요청이 처리되고 나서 실행됩니다
# 127.0.0.1 - - [13/Jul/2020 16:39:07] "GET / HTTP/1.1" 200 -

@app.teardown_request
def teardown_request(exception):
    print('[4] 매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다')

@app.teardown_appcontext
def teardown_appcontext(exception):
    print('[5] HTTP 요청의 애플리케이션 컨텍스트가 종료될 때 실행된다')
#  * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
# [1] 앱이 가동되고 나서 첫번째 HTTP 요청에만 응답합니다.
#     이 서버는 개인 자산이니 건들지 말것
#     곧 자료를 전송합니다
# [2] 매 HTTP 요청이 처리되기 전에 실행됩니다
# [★ ] app.route
# [3] 매 HTTP 요청이 처리되고 나서 실행됩니다
# [4] 매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다
# [5] HTTP 요청의 애플리케이션 컨텍스트가 종료될 때 실행된다
# 127.0.0.1 - - [13/Jul/2020 16:44:23] "GET / HTTP/1.1" 200 -

if __name__ == '__main__':
    app.run(host='127.0.0.1') # port의 default = 5000
'''
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
[1] 앱이 가동되고 나서 첫번째 HTTP 요청에만 응답합니다.
    이 서버는 개인 자산이니 건들지 말것
    곧 자료를 전송합니다
[2] 매 HTTP 요청이 처리되기 전에 실행됩니다
[★ ] app.route
[3] 매 HTTP 요청이 처리되고 나서 실행됩니다
[4] 매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다
[5] HTTP 요청의 애플리케이션 컨텍스트가 종료될 때 실행된다
127.0.0.1 - - [13/Jul/2020 16:52:19] "GET / HTTP/1.1" 200 -
[2] 매 HTTP 요청이 처리되기 전에 실행됩니다
[★ ] app.route
[3] 매 HTTP 요청이 처리되고 나서 실행됩니다
[4] 매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다
[5] HTTP 요청의 애플리케이션 컨텍스트가 종료될 때 실행된다
127.0.0.1 - - [13/Jul/2020 16:52:34] "GET / HTTP/1.1" 200 -
'''
