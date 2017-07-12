from encoder import Model
from matplotlib import pyplot as plt
from utils import sst_binary, train_with_reg_cv

model = Model()

trX, vaX, teX, trY, vaY, teY = sst_binary()
trXt = model.transform(trX)
vaXt = model.transform(vaX)
teXt = model.transform(teX)

# classification results
full_rep_acc, c, nnotzero = train_with_reg_cv(trXt, trY, vaXt, vaY, teXt, teY)
print('%05.2f test accuracy'%full_rep_acc)
print('%05.2f regularization coef'%c)
print('%05d features used'%nnotzero)

# visualize sentiment unit
sentiment_unit = trXt[:, 2388]
plt.hist(sentiment_unit[trY==0], bins=25, alpha=0.5, label='neg')
plt.hist(sentiment_unit[trY==1], bins=25, alpha=0.5, label='pos')
plt.legend()
plt.show()
