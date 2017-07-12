from encoder import Model
from utils import sst_binary, train_with_reg_cv

model = Model()

trX, vaX, teX, trY, vaY, teY = sst_binary()
trXt = model.transform(trX)
vaXt = model.transform(vaX)
teXt = model.transform(teX)

full_rep_acc, c, nnotzero = train_with_reg_cv(trXt, trY, vaXt, vaY, teXt, teY)
print('%05.2f test accuracy'%full_rep_acc)
print('%05.2f regularization coef'%c)
print('%05d features used'%nnotzero)
