import random
import math
import matplotlib.pyplot as plt

# Class of one bandit
class Bandit:
    id = 0
    mean = 0
    variance = 1

    def __init__(self, id, mean=0, variance=1):
        self.id = id
        self.mean = mean
        self.variance = variance

    def Update(self, mean=0, variance=1):
        self.mean += random.gauss(self.mean, self.variance)

    def Play(self):
        return random.gauss(self.mean, self.variance)

# Class of multi-armed bandit machine
# Manage multiple bandits
class Bandits:
    numbers = 0
    bandits = []
    stationary = True

    def __init__(self, numbers=10, mean=0, variance=1, stationary=True):
        for i in range(numbers):
            self.AddBandit(Bandit(i, random.gauss(mean, variance), 1))
        self.stationary = stationary

    def AddBandit(self, bandit):
        if type(bandit).__name__ == "Bandit":
            self.bandits.append(bandit)
            self.numbers += 1

    def Update(self, mean=0, variance=1):
        if not self.stationary:
            for bandit in self.bandits:
                bandit.Update(mean, variance)

    def Play(self, index):
        return self.bandits[index].Play()

    def OptimalReward(self):
        value = None
        optimal = 0
        for b in self.bandits:
            if value is None or b.mean > value:
                value = b.mean
                optimal = b
        
        return value, optimal

    def ResetMean(self, mean=0, variance=1):
        for b in self.bandits:
            b.mean = random.gauss(mean, variance)

# Epsilon Greedy or Greedy with/without constant step size
def EpiGreedy(bandits, epsilon=0, alpha=0, steps=1000, initEstimate=0):
    Q = [initEstimate] * bandits.numbers
    N = [0] * bandits.numbers

    rewardSum = 0
    rewardAverage = 0
    averageTrack = []
    optimalAction = 0
    optimalTrack = []

    for i in range(0, steps):
        # Select according to epsilon
        rand = random.random()
        if rand < epsilon:
            j = random.randint(0, bandits.numbers - 1)
        else:
            j = Q.index(max(Q))

        N[j] += 1
        r = bandits.Play(j)

        if(alpha == 0):
            Q[j] += (r - Q[j]) / N[j]
        else:
            Q[j] += (r - Q[j]) * alpha

        # Record Data
        rewardSum += r
        rewardAverage = rewardSum / (i + 1)
        averageTrack.append(rewardAverage)
        if j == bandits.OptimalReward()[1].id:
            optimalAction += 1
        optimalTrack.append(optimalAction / (i + 1))

        bandits.Update()

    return rewardAverage, averageTrack, optimalAction / steps, optimalTrack

# Upper Confidence Bound
def UCBGreedy(bandits, c=1, alpha=0, steps=1000, initEstimate=0):
    Q = [initEstimate] * bandits.numbers
    N = [0] * bandits.numbers

    rewardSum = 0
    rewardAverage = 0
    averageTrack = []
    optimalAction = 0
    optimalTrack = []

    for i in range(0, steps):
        j = 0
        maxValue = None
        for k in range(0, bandits.numbers):
            if N[k] == 0:
                j = k
                break

            if k == 0 or Q[k] + c * math.sqrt(math.log(i) / N[k]) > maxValue:
                j = k
                maxValue = Q[k] + c * math.sqrt(math.log(i) / N[k])

        N[j] += 1
        r = bandits.Play(j)

        if(alpha == 0):
            Q[j] += (r - Q[j]) / N[j]
        else:
            Q[j] += (r - Q[j]) * alpha

        # Record Data
        rewardSum += r
        rewardAverage = rewardSum / (i + 1)
        averageTrack.append(rewardAverage)
        if j == bandits.OptimalReward()[1].id:
            optimalAction += 1
        optimalTrack.append(optimalAction / (i + 1))

        bandits.Update()

    return rewardAverage, averageTrack, optimalAction / steps, optimalTrack

# Gradient Bandit
def GradientBandit(bandits, alpha=0.1, steps=1000, initEstimate=0):
    H = [initEstimate] * bandits.numbers
    P = [0] * bandits.numbers

    rewardSum = 0
    rewardAverage = 0
    averageTrack = []
    optimalAction = 0
    optimalTrack = []

    for i in range(1000):
        # Compute Probability
        PTotal = 0
        for j in range(bandits.numbers):
            PTotal += math.exp(H[j])
        for j in range(bandits.numbers):
            P[j] = math.exp(H[j]) / PTotal

        rand = random.random()

        selectIndex = -1
        for j in range(bandits.numbers):
            if rand > P[j]:
                rand -= P[j]
            else:
                selectIndex = j
                break

        r = bandits.Play(j)

        # Record Data
        rewardSum += r
        rewardAverage = rewardSum / (i + 1)
        averageTrack.append(rewardAverage)
        if j == bandits.OptimalReward()[1].id:
            optimalAction += 1
        optimalTrack.append(optimalAction / (i + 1))
        
        for j in range(bandits.numbers):
            if j == selectIndex:
                H[j] += alpha * (r - rewardAverage) * (1 - P[j])
            else:
                H[j] -= alpha * (r - rewardAverage) * P[j]

        bandits.Update()

    return rewardAverage, averageTrack, optimalAction / steps, optimalTrack

# Main function
# Compare Six Methods in 100 runs
if __name__ == "__main__":
    bandits = Bandits()
    avg = [0] * 6
    opt = [0] * 6

    for i in range(100):
        r1 = EpiGreedy(bandits)
        r2 = EpiGreedy(bandits, 0.1)
        r3 = EpiGreedy(bandits, 0.01)
        r4 = UCBGreedy(bandits)
        r5 = GradientBandit(bandits)
        r6 = EpiGreedy(bandits, 0, 0.1, 1000, 5)

        avg[0] += r1[0]
        avg[1] += r2[0]
        avg[2] += r3[0]
        avg[3] += r4[0]
        avg[4] += r5[0]
        avg[5] += r6[0]
        opt[0] += r1[2]
        opt[1] += r2[2]
        opt[2] += r3[2]
        opt[3] += r4[2]
        opt[4] += r5[2]
        opt[5] += r6[2]

        bandits.ResetMean()

    print("Epsilon 0 --> Reward: " + str(avg[0] / 100))
    print("Epsilon 0 --> Optimal Percentage: " + str(opt[0] / 100) + "\n")
    print("Epsilon 0.1 --> Reward: " + str(avg[1] / 100))
    print("Epsilon 0.1 --> Optimal Percentage: " + str(opt[1] / 100) + "\n")
    print("Epsilon 0.01 --> Reward: " + str(avg[2] / 100))
    print("Epsilon 0.01 --> Optimal Percentage: " + str(opt[2] / 100) + "\n")
    print("UCB Greedy with c = 1 --> Reward: " + str(avg[3] / 100))
    print("UCG Greedy with c = 1 --> Optimal Percentage: " + str(opt[3] / 100) + "\n")
    print("Gradient Bandit with a = 0.1 --> Reward: " + str(avg[4] / 100))
    print("Gradient Bandit with a = 0.1 --> " + str(opt[4] / 100) + "\n")
    print("Greedy with initial value = 5 --> Reward: " + str(avg[5] / 100))
    print("Greedy with initial value = 5 --> Optimal Percentage: " + str(opt[5] / 100) + "\n")



