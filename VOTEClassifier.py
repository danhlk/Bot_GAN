#from scipy.sparse.construct import rand
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, svm
from sklearn.model_selection import train_test_split, KFold
import numpy as np

class VOTEClassifier():
    def __init__(self) -> None:
        self.RF = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        self.SVM = svm.SVC()
        self.LR = linear_model.LogisticRegression()
        self.MLP = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, alpha=1e-1,
                                solver='sgd', verbose=0, tol=1e-4, random_state=1,
                                learning_rate_init=.1)
        self.GBC = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, random_state=1)
        self.alpha = 0.2 * np.ones((5, ))

    def fit(self, X, y):
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
        self.RF.fit(xtrain, ytrain)
        self.SVM.fit(xtrain, ytrain)
        self.LR.fit(xtrain, ytrain)
        self.GBC.fit(xtrain, ytrain)
        self.MLP.fit(xtrain, ytrain)
        
        RF_score = self.RF.score(xtest, ytest)
        SVM_score = self.SVM.score(xtest, ytest)
        LR_score = self.LR.score(xtest, ytest)
        GBC_score = self.GBC.score(xtest, ytest)
        MLP_score = self.MLP.score(xtest, ytest)
        self.alpha = np.array([RF_score, SVM_score, LR_score, GBC_score, MLP_score]) ** 5
        self.alpha = self.alpha / np.sum(self.alpha)
    
    def predict(self, X):
        RF_predict = self.RF.predict(X)
        SVM_predict = self.SVM.predict(X)
        LR_predict = self.LR.predict(X)
        GBC_predict = self.GBC.predict(X)
        MLP_predict = self.MLP.predict(X)
        VOTE_predict = self.alpha[0]*RF_predict + self.alpha[1]*SVM_predict + self.alpha[2]*LR_predict + \
                        self.alpha[3]*GBC_predict + self.alpha[4]*MLP_predict
        VOTE_predict = np.ones(VOTE_predict.shape)*(VOTE_predict > 0.5)
        return VOTE_predict

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=None)