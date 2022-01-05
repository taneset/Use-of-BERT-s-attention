

#%%Fig1
import matplotlib.pyplot as plt
import numpy as np
import os

labels = [ 'full-text', 'top-%50','bottom-%50']

bert= [93.4,91.6,80.1]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()

rects2 = ax.bar(x , bert,  color=[ 'blue','green','red'],width=[1.2,0.6,0.6])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy of Fine Tuned Model')
ax.set_xlabel('Fine Tuning Length')
ax.set_title('Accuracy of High and Low Attentions')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(78,95)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



autolabel(rects2)

fig.tight_layout()

plt.show()
plt.savefig(os.path.join('filt.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures': 23}}



# %% fig2

import matplotlib.pyplot as plt
import numpy as np


labels = [ '%6','%11.5','%23.5', '%47','full text']
sbert = [86.5,90.9, 91.9, 92.6,93.4]
bert= [80.0,85.4,86.6,91.5,93.4]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, sbert, width, label='SBERT')
rects2 = ax.bar(x + width/2, bert, width, label='BERT')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by filtering Bert\'s 1th layer and SBERT')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(75,100)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
plt.savefig(os.path.join('filt.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures': 23}}

# %%alternative
#%%
import matplotlib.pyplot as plt
D={ u"full-length":93.2 ,u"top-%50": 90.7, u"bottom-%50":80.1}
ax = plt.gca()
ax.set_ylim([75, 95])
bars=plt.bar(range(len(D)), list(D.values()), align='center',color=[ 'blue','green','red'],width=[1.2,0.6,0.6])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, yval)
plt.xticks(range(len(D)), list(D.keys()))
# Add title and axis names
plt.title('Accuracy Comparison of High and Low Attention')
plt.xlabel('Filtered Tokens')
plt.ylabel('Accuracy of Fine-tuned Model on Test Data')
 
# Create names on the x axis

plt.show()




# %%
