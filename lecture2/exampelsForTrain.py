import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("train.csv")


def importantFeatures():

    x = data.iloc[:, 0:20]
    x = x.fillna(x.median())
    y = data.iloc[:, -1]
    y = y.fillna(y.median())

    model = ExtraTreesClassifier()
    model.fit(x, y)
    print(model.feature_importances_)

    feat_importances = pd.Series(model.feature_importances_, index=x.columns)
    feat_importances.nlargest(5).plot(kind='barh')
    plt.show()


def strongCorrelation():
    x = data.iloc[:, 0:20]
    x = x.fillna(x.median())
    y = data.iloc[:, -1]
    y = y.fillna(y.median())

    bestfeatures = SelectKBest(score_func=chi2, k=5)

    fit = bestfeatures.fit(x, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x.columns)

    scores = pd.concat([dfcolumns, dfscores], axis=1)
    scores.columns = ['Specs', 'Score']
    print(scores.nlargest(9, 'Score'))
    print("\n",scores)

def selectFeaturesMostCorrelatedToTarget():
    x = data.iloc[:, 0:20]
    y = data.iloc[:, -1]

    correlation_matrix = data.corr()
    top_corr_features = correlation_matrix.index
    plt.figure(figsize=(20, 20))
    g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()

def main():
    keep_going = True
    while keep_going:
        print("1. Important Features")
        print("2. Strong Correlation")
        print("3. Select Features Most Correlated To Target")
        print("4. Exit")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            importantFeatures()
        elif choice == 2:
            strongCorrelation()
        elif choice == 3:
            selectFeaturesMostCorrelatedToTarget()
        elif choice == 4:
            keep_going = False
        else:
            print("Wrong choice")

main()