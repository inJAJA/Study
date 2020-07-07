import tensorflow as tf

node2 = tf.constant(2.0)
node3 = tf.constant(3.0)
node4 = tf.constant(4.0)
node5 = tf.constant(5.0)

# 3 + 4 + 5
add = tf.add_n([node3, node4, node5]) # 많은 양의 tesor한번에 처리 

# 4 - 3
sub = tf.subtract(node4, node3)       

# 3 * 4
mul  = tf.multiply(node3, node4)       
# mul2 = tf.matmul()                  # 행렬의 곱셈

# 4 / 2
div = tf.div(node4, node2)
mod = tf.mod(node4, node2)            # 나눈 나머지 값

sess = tf.Session()

print('3 + 4 + 5 =', sess.run(add))   # 3 + 4 + 5 = 12.0
print('4 - 3 =', sess.run(sub))       # 4 - 3 = 1.0
print('3 * 4 =', sess.run(mul))       # 3 * 4 = 12.0
print('4 / 2 =', sess.run(div))       # 4 / 2 = 2.0
print('4 % 2 =', sess.run(mod))       # 4 % 2 = 0.0


print(sess.run(node3 + node4))


