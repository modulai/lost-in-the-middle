import matplotlib.pyplot as plt

position_of_document = [1, 10, 20]
accuracy = [0.67, 0.52, 0.59]
title = "20 Total Retrieved Documents"

plt.plot(position_of_document, accuracy, "o-", label="gpt-3.5-turbo-0613 (open book)")
plt.title(title)
plt.legend()
plt.xlabel("Position of Document with the Answer")
plt.ylabel("Accuracy")
plt.savefig("Figure 1.png")
