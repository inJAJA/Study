weight = 0.5           # 초기 가중치
input = 0.5            # x
goal_prediction = 0.8  # y

lr = 0.05

for iteration in range(5):                           # 0.5를 넣어서 0.8을 찾아가는 과정
    prediction = input*weight                           # y = w*x 
    error = (prediction - goal_prediction)**2           # loss

    print('Error : ' + str(error)+'\tPrediction : '+str(prediction))

    up_prediction = input *(weight + lr)                # weight = gradient : -기울기 +올림
    up_error = (goal_prediction - up_prediction)**2     # loss

    down_predicrion = input*(weight - lr)               # weight = gradient : +기울기 -내림
    down_error = (goal_prediction - down_predicrion)**2 # loss


    # dwon_error와 up_error를 비교하여 error값이 작은 쪽으로 새롭게 weigt갱신
    if(down_error < up_error):                          
        weight = weight - lr                                            

    if(down_error > up_error):                          
        weight = weight + lr 

                               