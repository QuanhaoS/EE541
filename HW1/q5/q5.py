# Question 5
import random
import matplotlib.pyplot as plt

def toss_coin(p=0.7):
    return 1 if random.random() < p else 0

def longest_run_of_heads(trials):
    max_run = 0
    current_run = 0
    for t in trials:
        if t == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run

trials = [toss_coin() for _ in range(50)]

num_heads = sum(trials)
longest_run = longest_run_of_heads(trials)

print("Number of heads:", num_heads)
print("Longest run of heads:", longest_run)

def repeated_experiments(num_experiments, num_flips=50):
    counts = []
    for _ in range(num_experiments):
        trials = [toss_coin() for _ in range(num_flips)]
        counts.append(sum(trials))
    return counts

experiment_sizes = [20, 100, 200, 1000]

for n in experiment_sizes:
    counts = repeated_experiments(n)

    plt.figure()
    plt.hist(counts, bins=10)
    plt.title(f"{n} experiments of 50 coin tosses")
    plt.xlabel("Number of heads")
    plt.ylabel("Frequency")
    plt.show()

def head_run_lengths(trials):
    runs = []
    current_run = 0
    for t in trials:
        if t == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
                current_run = 0
    if current_run > 0:
        runs.append(current_run)
    return runs

trials = [toss_coin() for _ in range(500)]
runs = head_run_lengths(trials)

plt.figure()
plt.hist(runs, bins=range(1, max(runs) + 2))
plt.xlabel("Run length of heads")
plt.ylabel("Frequency")
plt.title("Histogram of head run lengths (500 tosses)")
plt.show()
