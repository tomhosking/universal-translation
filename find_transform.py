import json,random
import numpy as np
import matplotlib.pyplot as plt

def find_nearest_ix(array,value):
    diff = np.sum(np.square(array-value), axis=1)
    idx = diff.argmin()
    return idx
def find_nearest_k_ix(array,value,k):
    diff = np.sum(np.square(array-value), axis=1)
    idx = diff.argpartition([i for i in range(k)])
    return idx[:k]

with open('embeddings_20_0.json', 'r') as outfile:
    data1 = json.load(outfile)
with open('embeddings_20_1.json', 'r') as outfile:
    data2 = json.load(outfile)

for k in data1['dictionary']:
    if k not in data2['dictionary'].keys():
        print('Missing word: ', k)



data1['embeddings'] = np.array(data1['embeddings'])
data2['embeddings'] = np.array(data2['embeddings'])

accuracies=[]
top_5_accuracies=[]
for num_indices in range(20,100,10):
    print('Fitting to ',num_indices,' words')
    indices = [random.randint(0,len(data1['dictionary'])-1) for i in range(num_indices)]
    test_indices = [random.randint(0,len(data1['dictionary'])-1) for i in range(10)]

    embeddings1 = data1['embeddings'][indices,:]
    embeddings2 = data2['embeddings'][indices,:]

    transform,_,_,_ = np.linalg.lstsq(embeddings1,embeddings2)
    # transform = transform/np.abs(np.linalg.det(transform))

    embeddings2_est = data1['embeddings'].dot(transform)

    for ix in test_indices:
        src_wrd = data1['reverse_dictionary'][str(ix)]
        tgt_ix = find_nearest_ix(data2['embeddings'], embeddings2_est[ix,:])
        tgt_wrd = data2['reverse_dictionary'][str(tgt_ix)]

        top_5 = find_nearest_k_ix(data2['embeddings'], embeddings2_est[ix,:],5)
        top_5_words = [data2['reverse_dictionary'][str(ix)] for ix in top_5]
        print(src_wrd, tgt_wrd, top_5_words)

    is_correct=[]
    is_close=[]
    for i in range(len(data1['reverse_dictionary'])):
        src_wrd = data1['reverse_dictionary'][str(i)]
        tgt_ix = find_nearest_ix(data2['embeddings'], embeddings2_est[i,:])
        tgt_wrd = data2['reverse_dictionary'][str(tgt_ix)]

        top_5 = find_nearest_k_ix(data2['embeddings'], embeddings2_est[i,:],5)
        top_5_words = [data2['reverse_dictionary'][str(ix)] for ix in top_5]
        is_correct.append(1. if src_wrd==tgt_wrd else 0.)
        is_close.append(1. if src_wrd in top_5_words else 0.)

    accuracies.append(np.mean(is_correct))
    top_5_accuracies.append(np.mean(is_close))
    # print(top_5_accuracies)

plt.title('Reconstruction accuracy for transformed embeddings')
plt.xlabel('Samples used for estimation')
plt.ylabel('Accuracy')
plt.legend()
plt.plot([x for x in range(20,100,10)], accuracies,label='Top-1')
plt.plot([x for x in range(20,100,10)], top_5_accuracies,label='Top-5')
plt.show()
