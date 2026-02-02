# Question 1
W1 = [
    [1, -2],
    [3,  4]
]
b1 = [1, 0]

W2 = [
    [2,  2],
    [2, -3]
]
b2 = [0, -4]

x = [1, -1]

# Matrix-vector multiplication
def mat_vec_mul(W, v):
    return [sum(W[i][j] * v[j] for j in range(len(v)))
            for i in range(len(W))]

# Vector-vector addition
def vec_add(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

# Vector-vector subtraction
def vec_sub(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]

# Vector-scalar multiplication
def vec_mul(v, s):
    return [v[i] * s for i in range(len(v))]

# Forward pass
def forward_pass(x):
    return vec_add(mat_vec_mul(W1, x), b1), mat_vec_mul(W2, x) + b2

# Backward pass
def backward_pass(x, y):
    return vec_sub(y, x), vec_mul(x, 2)

# Update weights
def update_weights(W, b, dW, db):
    return vec_add(W, vec_mul(dW, 0.01)), vec_add(b, vec_mul(db, 0.01))

def relu(v):
    return [max(0, x) for x in v]

# Forward
z1 = vec_add(mat_vec_mul(W1, x), b1)
a1 = relu(z1)

z2 = vec_add(mat_vec_mul(W2, a1), b2)
a2 = z2

print("Hidden layer output a1:", a1)
print("Final output a2:", a2)