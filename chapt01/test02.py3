import tensorflow as tf

#构造OP
matrix1=tf.constant([[3.,3.]])

matrix2=tf.constant([[2.0],[2.0]])

product=tf.matmul(matrix1,matrix2)

#创建session会话，隐式关闭session，调用sess.close()
with tf.Session() as sess:
	result=sess.run(product)
	print(result)
