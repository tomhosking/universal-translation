import json,random
import numpy as np
import matplotlib.pyplot as plt

def find_nearest_ix(array,value):
    diff = np.sum(np.square(array-value), axis=1)
    idx = diff.argmin()
    return idx

with open('embeddings_32_1.json', 'r') as outfile:
    data1 = json.load(outfile)
with open('embeddings_32_2.json', 'r') as outfile:
    data2 = json.load(outfile)

for k in data1['dictionary']:
    if k not in data2['dictionary'].keys():
        print('Missing word: ', k)



data1['embeddings'] = np.array(data1['embeddings'])
data2['embeddings'] = np.array(data2['embeddings'])

accuracies=[]

for num_indices in range(1,1000):
    indices = [random.randint(0,len(data1['dictionary'])-1) for i in range(num_indices)]
    test_indices = [random.randint(0,len(data1['dictionary'])-1) for i in range(10)]

    embeddings1 = data1['embeddings'][indices,:]
    embeddings2 = data2['embeddings'][indices,:]

    transform,_,_,_ = np.linalg.lstsq(embeddings1,embeddings2)

    embeddings2_est = data1['embeddings'].dot(transform)

    for ix in test_indices:
        src_wrd = data1['reverse_dictionary'][str(ix)]
        tgt_ix = find_nearest_ix(embeddings2_est, data2['embeddings'][ix,:])
        tgt_wrd = data2['reverse_dictionary'][str(tgt_ix)]
        # print(src_wrd, tgt_wrd)

    is_correct=[]
    for i in range(len(data1['reverse_dictionary'])):
        src_wrd = data1['reverse_dictionary'][str(i)]
        tgt_ix = find_nearest_ix(embeddings2_est, data2['embeddings'][i,:])
        tgt_wrd = data2['reverse_dictionary'][str(tgt_ix)]
        is_correct.append(1. if src_wrd==tgt_wrd else 0.)

    accuracies.append(np.mean(is_correct))

plt.plot([x for x in range(1,1000)], accuracies)
plt.show()
