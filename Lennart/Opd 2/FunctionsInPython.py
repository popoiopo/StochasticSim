def averageQueueTimesTwee(lamb, mu):
    up = 2 * (lamb / mu) ** 3
    low = lamb * (1 - (lamb / mu) ** 2)
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

print("For 2 servers using variant of M/M/2")
print(f"Average time in queue calculated = {averageQueueTimesTwee(12, 13)}")
print(f"Average time calculated = {averageTimesTwee(12, 13)}")
print(f"Average customers in queue calculated = {averageQueueCustomersTwee(12, 13)}")
print(f"Average customers calculated = {averageCustomersTwee(12, 13)}\n")



def averageQueueTimesOne(lamb, mu):
    rho = lamb / mu
    up = rho ** 2
    low = lamb * (1 - rho)
    return up / low

print("For 1 servers")
print(f"Average time in queue calculated = {averageQueueTimesOne(12, 13)}")