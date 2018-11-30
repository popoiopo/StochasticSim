import math

def averageQueueTimesTwee(lamb, mu):
    rho = lamb/mu
    up = (rho ** 3)
    low = lamb * (1 - (rho ** 2))
    return up / low

def averageTimesTwee(lamb, mu):
    up = mu ** 2 + lamb ** 2
    low = mu * (mu ** 2 - lamb ** 2)
    return up / low

def averageCustomersTwee(lamb, mu):
    rho = lamb / mu
    up = rho * (mu ** 2 + lamb ** 2)
    low = mu ** 2 - lamb ** 2
    return up / low

def averageQueueCustomersTwee(lamb, mu):
    rho = lamb / mu
    up = 2 * rho ** 3
    low = 1 - rho ** 2
    return up / low



def C(rho, n):
    up = ((rho ** n) / math.factorial(n)) * (n / (n - rho))
    low1 = 0
    for k in range(n):
        low1 += ((rho ** k) / math.factorial(k))
    low2 = ((rho ** n) * n) / (math.factorial(n) * (n - rho))
    low = low1 + low2
    return up / low

def ErlangC(A, N):
    L = (A**N / math.factorial(N)) * (N / (N - A))
    sum_ = 0
    for i in range(N):
        sum_ += (A**i) / math.factorial(i)
    return (L / (sum_ + L))

def Wq(rho, mu, n):
    Pq = C(rho, n)
    a = 1 / (mu * (n - rho))
    return a * Pq

l = 0.95
u = 1.0
rho = l/u

print(C(rho, 1) == rho)


print("For 2 servers using variant of M/M/2")
print(f"Average time in queue calculated = {averageQueueTimesTwee(l, u)}\n")
# print(f"Average time calculated = {averageTimesTwee(l, u)}")
# print(f"Average customers in queue calculated = {averageQueueCustomersTwee(l, u)}")
# print(f"Average customers calculated = {averageCustomersTwee(l, u)}\n")



def averageQueueTimesOne(lamb, mu):
    rho = lamb / mu
    up = rho
    low = mu * (1 - rho)
    return up / low

print("For 1 servers")
print(f"Average time in queue calculated = {averageQueueTimesOne(l, u)}")

n=1
l = 0.96
u = 1.0
rho = l / u
print("\nFor n=1")
print(f"C(rho, 2) = {C(rho, n)}")
print(f"Wq = {Wq(rho, u, n)}")
print(f"Wq variant = {Wq(n * rho, u, n)}")

n=2
print("\nFor n=2")
print(f"C(rho, 2) = {C(rho, n)}")
print(f"Wq = {Wq(rho, u, n)}")
print(f"Wq variant = {Wq(n * rho, u, n)}")

n=3
print("\nFor n=3")
print(f"C(rho, 2) = {C(rho, n)}")
print(f"Wq = {Wq(rho, u, n)}")
print(f"Wq variant = {Wq(n * rho, u, n)}")

n=4
print("\nFor n=4")
print(f"C(rho, 2) = {C(rho, n)}")
print(f"Wq = {Wq(rho, u, n)}")
print(f"Wq variant = {Wq(n * rho, u, n)}")
