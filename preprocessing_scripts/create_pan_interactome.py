# %%
import numpy as np

human_PAN = np.load("npy_file_new(human_dataset).npy")

positive_int = []
negative_int = []

for np1,uni1,pdb1,np2,uni2,pdb2,label in human_PAN:
    if int(label) == 1:
        positive_int.append((uni1,uni2))
    negative_int.append((uni1,uni2))

with open("pan_human_postive.txt", "w") as f:
    f.write("protein1 protein2\n")
    for protein1, protein2 in positive_int:
        f.write(f"{protein1} {protein2}\n")

with open("pan_human_negative.txt", "w") as f:
    f.write("protein1 protein2\n")
    for protein1, protein2 in negative_int:
        f.write(f"{protein1} {protein2}\n")
