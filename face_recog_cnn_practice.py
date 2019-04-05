import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from dataload import load_data

#CNN 모델 - 2차원 평면
DATA_DIR = './data/'
data = load_data(DATA_DIR)

X = tf.placeholder(tf.float32, [None,32,32,1])
Y = tf.placeholder(tf.float32, [None, 22]) 
# drop out 비율 
keep_prob = tf.placeholder(tf.float32)

# 계층 1 정의 # W1 [3 3 1 32] -> [3 3]: 커널 크기, 1: 입력값 X 의 특성수, 32: 필터 갯수
#  # L1 Conv shape=(?, 28, 28, 32) 
# # Pool ->(?, 14, 14, 32) 
# # Convolution 

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) 
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') 
L1 = tf.nn.relu(L1) 

# Pooling 
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 계층 2 정의 
# L2 Conv shape=(?, 14, 14, 64) 
# Pool ->(?, 7, 7, 64)

# Convolution 
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) 
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') 
L2 = tf.nn.relu(L2) 

# Pooling 
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 완전 연결 레이어: 입력값 7x7x64 -> 출력값 256 
# 직전의 Pool 사이즈인 (?, 7, 7, 64) 를 참고하여 차원을 줄임 
# Reshape ->(?, 256) 

W3 = tf.Variable(tf.random_normal([8 * 8 * 64, 256], stddev=0.01)) 
L3 = tf.reshape(L2, [-1, 8 * 8 * 64]) 
L3 = tf.matmul(L3, W3) 
L3 = tf.nn.relu(L3) 

L3 = tf.nn.dropout(L3, keep_prob)

#최종출력값
W4 = tf.Variable(tf.random_normal([256, 22], stddev=0.01)) 
model = tf.matmul(L3, W4)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y)) 

#optimizer = tf.train.AdamOptimizer(0.001).minimize(cost) 

# 최적화 함수로 RMSPropOptimizer를 사용하는 경우 
optimizer = tf.train.AdamOptimizer(0.001, 0.9).minimize(cost)


init = tf.global_variables_initializer() 
sess = tf.Session() 
sess.run(init) 
batch_size = 1
total_batch = int(data.train.num_examples / batch_size) 
for epoch in range(50): 
    total_cost = 0 
    for i in range(total_batch): 
        batch_xs, batch_ys = data.train.next_batch(batch_size) 
        
       # 이미지 데이터를 CNN 모델을 위핚 자료형태인 [28 28 1] 의 형태로 재구성 
        batch_xs = batch_xs.reshape(-1, 32, 32, 1)
        #print("batch_xs = ", batch_xs)
        _, cost_val = sess.run([optimizer, cost],
            feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7}) 
        total_cost += cost_val

    data.train.reset_batch()
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch)) 
print('최적화 완료!') 
        
# 결과 확인 
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1)) 
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) 
print('정확도:', sess.run(accuracy, feed_dict={X: data.test.images.reshape(-1, 32, 32, 1), 
    Y: data.test.labels, keep_prob: 1}))