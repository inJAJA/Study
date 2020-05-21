# Study
======== LSTM ==========
![LSTM_model](https://user-images.githubusercontent.com/64456018/82515517-56425500-9b53-11ea-9a03-d0cce4b12a0f.png)
![Processing sequence one by one](https://user-images.githubusercontent.com/64456018/82515670-bdf8a000-9b53-11ea-8b32-c4222d2c375d.gif)
< Processing sequence one by one >
![RNN_Next](https://user-images.githubusercontent.com/64456018/82515522-5b9f9f80-9b53-11ea-89c0-d597cbfa7d94.gif)
< h(t)가 다음 Time_step으로 넘어간다 : Passing hidden state to next time step >
![RNN작동](https://user-images.githubusercontent.com/64456018/82515525-5d696300-9b53-11ea-8e68-f539a8089628.gif)
<RNN Cell : 이전 입력 h(t-1) + 현재 입력 x(t) >
![LSTM_ForgetGate](https://user-images.githubusercontent.com/64456018/82515529-5fcbbd00-9b53-11ea-957b-58cfdc0c9f07.gif)
< Forget gate >
![LSTM_InputGate](https://user-images.githubusercontent.com/64456018/82515533-61958080-9b53-11ea-915f-d15f4e58feff.gif)
< Input Gate > 
![LSTM_CellGate](https://user-images.githubusercontent.com/64456018/82515536-63f7da80-9b53-11ea-9436-2d15b6f2678a.gif)
< Cell State >
![LSTM_OutputGate](https://user-images.githubusercontent.com/64456018/82515540-665a3480-9b53-11ea-82ec-9fc880ee9b71.gif)
< Output Gate > 

![sigmoid](https://user-images.githubusercontent.com/64456018/82516748-850dfa80-9b56-11ea-8433-9ab4961d6411.gif)
< sigmoid  :Sigmoid 함수는 모든 실수 입력 값을 0보다 크고 1보다 작은 미분 가능한 수로 변환하는 특징>

![tanh](https://user-images.githubusercontent.com/64456018/82516746-83443700-9b56-11ea-942a-85d6a0895b6c.gif)
< tanh : tanh 함수는 값이 -1과 1 사이에 있도록하여 신경망의 출력을 조절

출처 : https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21
