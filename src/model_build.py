
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os
import pickle
import sys
def build(X_train,X_test,y_train,y_test):
    model=RandomForestRegressor(n_estimators=300)

    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    r2score=r2_score(y_pred,y_test)
    print(r2score)

    os.makedirs("models",exist_ok=True)
    with open("models/model.pkl","wb") as f:
        pickle.dump(model,f)

    return y_pred,r2score

