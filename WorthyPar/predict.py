import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split as TTS

if __name__ == '__main__':
    data = pd.read_csv('./output.csv')
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    Xtrain, Xtest, Ytrain, Ytest = TTS(X, Y, test_size=0.1, random_state=420)

    reg_mod = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        subsample=0.75,
        colsample_bytree=1,
        max_depth=7,
        gamma=0,
    )

    eval_set = [(Xtrain, Ytrain), (Xtest, Ytest)]
    reg_mod.fit(Xtrain, Ytrain, eval_set=eval_set, eval_metric='rmse', verbose=False)

    scores = cross_val_score(reg_mod, Xtrain, Ytrain, cv=10)
    print("Mean cross-validation score: %.2f" % scores.mean())

    predictions = reg_mod.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(Ytest, predictions))
    print("RMSE: %f" % (rmse))
    r2 = np.sqrt(r2_score(Ytest, predictions))
    print("R_Squared Score : %f" % (r2))

    # Loss
    sns.set_style("white")
    palette = sns.color_palette("Set2", n_colors=2)
    plt.plot(reg_mod.evals_result()['validation_0']['rmse'], label='train', color=palette[0], linewidth=2)
    plt.plot(reg_mod.evals_result()['validation_1']['rmse'], label='test', color=palette[1], linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('Loss.png')
    plt.show()

    # Fitting
    sns.set_style("white")
    palette = sns.color_palette("husl", n_colors=2)
    x_ax = range(len(Ytest))
    plt.plot(x_ax, Ytest, label="True Values", color=palette[0], linewidth=1)
    plt.plot(x_ax, predictions, label="Predicted Values", color=palette[1], linewidth=1)
    plt.xlabel("Sample Number")
    plt.ylabel("Scan Size")
    plt.legend()
    plt.savefig('True vs Predicted Values.png')
    plt.show()
