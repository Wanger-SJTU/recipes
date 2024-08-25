
import numpy as np
import random

# powerinfer 中计算是顺序无所谓的

input = np.random.random(size=[1, 4096])
up = np.random.random(size=[4096, 14336])
gate = np.random.random(size=[4096, 14336])
down = np.random.random(size=[14336, 4096])
mask = np.random.random(size=[1, 14336]) > 0.999

golden = np.matmul(np.matmul(input, gate * mask), down * mask.reshape(14336, 1))
size = mask.sum()
res = np.zeros_like(input)

masked_gate_ori = gate[:,mask[0]]
masked_gate = masked_gate_ori.copy().transpose(1,0)
masked_gate = masked_gate.transpose(1,0)
masked_down = down[mask[0],:].copy()


idx = [i for i in range(size)]
random.shuffle(idx)
for i in (idx):
    res += np.dot(masked_gate[:,i], input[0]) * masked_down[i]
print((golden-res).sum())