import tensorflow as tf 
from dataload import load_data



DATA_DIR = './data/'
data = load_data(DATA_DIR)


# 입력 데이터를 위핚 플레이스홀더, 가중치
 
X = tf.placeholder(tf.float32, [None, 1024]) #1024
Y = tf.placeholder(tf.float32, [None, 22]) #42

W1 = tf.Variable(tf.random_normal([1024, 256], stddev=0.01)) 
L1 = tf.nn.relu(tf.matmul(X, W1)) 
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01)) 
L2 = tf.nn.relu(tf.matmul(L1, W2)) 
W3 = tf.Variable(tf.random_normal([256, 22], stddev=0.01)) 
model = tf.matmul(L2, W3) 
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y)) 
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


init = tf.global_variables_initializer() 
sess = tf.Session()
sess.run(init)

batch_size = 40
total_batch = int(data.train.num_examples/batch_size)

for epoch in range(100):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = data.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={
            X :batch_xs,
            Y: batch_ys
            })
        total_cost += cost_val
    
    #batch 인덱스 초기화

    data.train.reset_batch()

    print('Epoch:', '%04d' % (epoch + 1), 
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')


# 결과 확인 
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1)) 
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) 
print('정확도:', sess.run(accuracy,
    feed_dict={X: data.test.images,
    Y: data.test.labels}))