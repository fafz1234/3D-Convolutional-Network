# http://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
test = pd.Series(np.random.choice(['w/d','dw','gct','lfa','hw'], 1000, p=[0.4, 0.1, 0.1, 0.2, 0.2]))
ax = test.value_counts(normalize = True).plot(kind='barh', figsize=(10,7),
                                        color="coral", fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Where were the battles fought?", fontsize=18)
ax.set_xlabel("Number of Battles", fontsize=18);
# ax.set_xticks([0, 250, 500, 750, 1000])
ax.set_xticks([0, .1, .2, .3, .4,.5])

totals = []
# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())
    
# set individual bar lables using above list
total = sum(totals)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.01, i.get_y()+.32, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=15, color='dimgrey')

ax.invert_yaxis()
test2 = ['w/d'] * 100
test2.extend(['dw'] *50)

####### vertical plot
# ax = test.value_counts().plot(kind='bar', figsize=(10,7),
#                                         color="coral", fontsize=13);

# ax.set_alpha(0.8)
# ax.set_title("Where were the battles fought?", fontsize=18)
# ax.set_ylabel("Number of Battles", fontsize=18);
# ax.set_yticks([0, 125, 250, 375, 500])

# # create a list to collect the plt.patches data
# totals = []

# # find the values and append to list
# for i in ax.patches:
#     totals.append(i.get_height())
# #     print i.get_height()

# # set individual bar lables using above list
# total = sum(totals)

# # set individual bar lables using above list
# for i in ax.patches:
#     # get_x pulls left or right; get_height pushes up or down
#     ax.text(i.get_x()+.07, i.get_height()+3.5, \
#             str(round((i.get_height()/float(total))*100, 2))+'%', fontsize=15,
#                 color='dimgrey')
