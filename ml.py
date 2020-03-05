import pandas as pd
dataset = pd.read_csv(r'C:\Users\meesum\Downloads\SimpleLR\Simple_Linear_Regression\Salary_Data.csv')
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
plt.scatter(X, y) 
plt.xlabel('x') 
plt.xlabel('y') 
plt.title("Training Data") 
plt.show() 
XP = tf.placeholder("float") 
YP = tf.placeholder("float") 
W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b") 
learning_rate = 0.01
training_epochs = 1000
n = len(X)
# Hypothesis 
y_pred = tf.add(tf.multiply(XP, W), b) 
  
# Mean Squared Error Cost Function 
cost = tf.reduce_sum(tf.pow(y_pred-YP, 2)) / (2 * n) 
  
# Gradient Descent Optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
  
# Global Variables Initializer 
init = tf.global_variables_initializer() 
# Starting the Tensorflow Session 
with tf.Session() as sess: 
      
    # Initializing the Variables 
    sess.run(init) 
      
    # Iterating through all the epochs 
    for epoch in range(training_epochs): 
          
        # Feeding each data point into the optimizer using Feed Dictionary 
        for (_x, _y) in zip(X, y): 
            sess.run(optimizer, feed_dict = {XP : _x, YP : _y}) 
          
        # Displaying the result after every 50 epochs 
        if (epoch + 1) % 50 == 0: 
            # Calculating the cost a every epoch 
            c = sess.run(cost, feed_dict = {XP : X, YP : y}) 
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 
      
    # Storing necessary values to be used outside the Session 
    training_cost = sess.run(cost, feed_dict ={XP: X, YP: y}) 
    weight = sess.run(W) 
    bias = sess.run(b) 
    # Calculating the predictions 
predictions = weight * X + bias 
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n') 
# Plotting the Results 
plt.plot(X, y, 'ro', label ='Original data') 
plt.plot(X, predictions, label ='Fitted line') 
plt.title('Linear Regression Result') 
plt.legend() 
plt.show() 
