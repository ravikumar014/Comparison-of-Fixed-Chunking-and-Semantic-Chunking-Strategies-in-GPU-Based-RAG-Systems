import matplotlib.pyplot as plt
import json

k_values = [1, 3, 5, 10]

fixed_recall = [0.2, 0.4, 0.5, 0.6]
semantic_recall = [0.3, 0.5, 0.5, 0.7]

plt.figure()

plt.plot(k_values, fixed_recall, marker='o', label="Fixed")
plt.plot(k_values, semantic_recall, marker='o', label="Semantic")

plt.xlabel("k")
plt.ylabel("Recall@k")
plt.title("Recall vs k Comparison")

plt.legend()

plt.savefig("results/plots/recall_vs_k.png")

plt.show()