import pandas as pd
import numpy as np

print("IU2141230160 - Anshu Patel\n")

data = pd.read_csv("traindata.csv")
print("Dataset:\n", data, "\n")

attributes = np.array(data)[:, :-1]
print("Attributes:\n", attributes)

target = np.array(data)[:, -1]
print("Target:\n", target)

def train(attributes, target):
    specific_hypothesis = None

    for i, val in enumerate(target):
        if val == "Yes":
            specific_hypothesis = attributes[i].copy()
            break

    if specific_hypothesis is not None:
        for i, val in enumerate(attributes):
            if target[i] == "Yes":
                for x in range(len(specific_hypothesis)):
                    if val[x] != specific_hypothesis[x]:
                        specific_hypothesis[x] = '?'
    return specific_hypothesis

final_hypothesis = train(attributes, target)
print("\nFinal hypothesis:\n", final_hypothesis)
