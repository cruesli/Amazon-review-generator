import re
import numpy as np
import matplotlib.pyplot as plt

f = open("Text generation/data.txt", mode="r").read()
liste = np.array(re.split(",| \n", f))

#loops through each of the model numbers and adds the loss numbers to the plot
for i in range(1,6):
    n_liste = liste[int(len(liste)/5)*(i-1): int(len(liste)/5)*i]

    loss = []
    for l in range(3, len(n_liste), 2):
        loss = np.append(loss, float(n_liste[l].split(": ")[1]))

    epo = np.arange(len(loss))
    plt.plot(epo, loss)


plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["1 Star", "2 Star", "3 Star", "4 Star", "5 Star"])
plt.show()