# 속성 3개
# 중간층
from sklearn.datasets import load_iris
iris=load_iris()
X=tf.constant(iris.data[:,[0,1,2]],dtype=tf.float32) #SepalLengthCm,SepalWidthCm,PetalLengthCm를 입력값 x로 함
y=tf.constant(iris.data[:,3],dtype=tf.float32)

w=tf.Variable(tf.random.normal([3,5]))
b=tf.Variable(tf.random.normal([5]))

u=tf.nn.relu(X@w+b) 
#인풋 값인 X와 가중치인 w 그리고 상수항이 b를 사용하여 비선형함수인 relu 함수를 사용하고 이를 통해
hidden layer를 활성화한다.

# 150x5
ww=tf.Variable(tf.random.normal([5,5]))
bb=tf.Variable(tf.random.normal([5]))

uu=tf.nn.relu(u@ww+bb)

www=tf.Variable(tf.random.normal([5,1]))
bbb=tf.Variable(tf.random.normal([]))

pred_y=uu@www+bbb 
#중간 레이어를 2번 거치면 학습 손실을 줄인 값고로 PetalWidthCm에 대한 예측 값이다.

mse=tf.reduce_mean(tf.square(y-pred_y)) 
#실제 값인 y와 오류 값인 pred_y 차인 오차를 제곱하여 평균을 구한 평균제곱오차를 mse에 넣음
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) 
train_op=optimizer.minimize(mse)  
#오류를 최소하하기 위하여 학습률을 0.001로 한 선형하강법을 한 결과를 담은 optimizer의 최소 값을 train_op로 삽입  

costs=[]

tf.global_variables_initializer().run()

for i in range(300):
    sess.run(train_op)
    costs.append(mse.eval())
plt.plot(costs)
