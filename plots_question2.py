import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

def sentence_lengths(filename):

    lines = [line.rstrip('\n') for line in open(filename)]


    sent_len = []

    for i in range(0,len(lines)):
        x = len(lines[i].split()) - 1
        sent_len.append(x)
        
    return sent_len



filename_en = '/home/claudia/Msc/NLU+/NLUplus_coursework2/nmt_toolkit/raw_data/train.en'
filename_jp = '/home/claudia/Msc/NLU+/NLUplus_coursework2/nmt_toolkit/raw_data/train.jp'

length_en = sentence_lengths(filename_en)
#print("English sentence length", length_en)

length_jp = sentence_lengths(filename_jp)
#print("Japanese sentence length", length_jp)

dims = (10,6)

fig, ax = plt.subplots(figsize=dims)
sns.distplot(length_jp, ax=ax, label='Japanese')
sns.distplot(length_en, ax=ax, label='English')
plt.title('Distribution of sentence lengths across Japanese and English text data')
ax.set(xlabel='Sentence length', ylabel='Density')
plt.legend()
plt.savefig('/home/claudia/Msc/NLU+/NLUplus_coursework2/Plots/lengths.png')

##correlation
print("Pearson Correlation Coefficient", np.corrcoef(length_jp,length_en))

##corr plot 

sns.regplot(np.array(length_jp), np.array(length_en))
plt.xlabel('English sentence length')
plt.ylabel('Japanese sentence length')
plt.title('Distribution of sentence lengths')
plt.savefig('/home/claudia/Msc/NLU+/NLUplus_coursework2/Plots/correlation.png')
