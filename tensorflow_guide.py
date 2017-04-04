import tensorflow as tf

# Create Session
session = tf.Session()

# Create Computes
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)

print(node1)
print(node2)

#Execute Static compute
session.run([node1, node2])

# Execute Dynamic Compute
node3 = tf.add(node1, node2)
session.run(node3)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder = a + b
session.run(adder, {a : 1, b: 2})
session.run(adder, {a : 7, b: 5})
