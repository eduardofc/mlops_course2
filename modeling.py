from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import time
import yaml


def read_config():
    with open('config.yaml', 'rb') as file:
        config = yaml.safe_load(file)
    return config

def set_train_test(feature_table, train_table, test_table, test_size, seed, model_name, model_version, exec_date):
    df = pd.read_csv(feature_table)
    feat_cols = [c for c in df.columns if c!='y']
    y = df['y']
    X = df[feat_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    df_train = pd.merge(X_train, y_train, left_index=True, right_index=True, how='inner')
    df_test = pd.merge(X_test, y_test, left_index=True, right_index=True, how='inner')

    df_train['model_name'] = model_name
    df_test['model_name'] = model_name
    df_train['model_version'] = model_version
    df_test['model_version'] = model_version
    df_train['exec_date'] = exec_date
    df_test['exec_date'] = exec_date

    df_train.to_csv(train_table, index=None)
    df_test.to_csv(test_table, index=None)

    return df_train, df_test, feat_cols

def modeling(df_train, df_test, feat_cols, seed):

    X_train = df_train[feat_cols]
    y_train = df_train['y']
    X_test = df_test[feat_cols]
    y_test = df_test['y']

    model = LogisticRegression(random_state=seed)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score

def log_metrics(table_metrics, model_name, model_version, score, exec_date):
    df_metrics = pd.DataFrame([[model_name, model_version, score, exec_date]],
                          columns=['model_name', 'model_version', 'score', 'exec_date'])
    df_metrics.to_csv(table_metrics, index=None)

if __name__=="__main__":

    # fichero de configuracion
    model_name = "iris"
    model_version = 1
    config = read_config()
    seed = config['modeling']['seed']
    test_size = config['modeling']['test_size']
    feature_table = config['tables']['table_features']
    feature_train = config['tables']['table_train']
    feature_test = config['tables']['table_test']
    feature_metrics = config['tables']['table_metrics']

    # fecha de ejecucion
    fecha_ejecucion = time.localtime()
    fecha_ejecucion = time.strftime("%Y-%m-%dT%H:%M:%S", fecha_ejecucion)
    print(f"Fecha Ejecucion: {fecha_ejecucion}")

    # train y test tables
    df_train, df_test, feat_cols = set_train_test(
        feature_table=feature_table,
        train_table=feature_train,
        test_table=feature_test,
        test_size=test_size,
        seed=seed,
        model_name=model_name,
        model_version=model_version,
        exec_date=fecha_ejecucion
    )

    # modelización
    model, score = modeling(df_train, df_test, feat_cols, seed)
    print(f"Model score: {score:.2f}")

    # log metrics
    log_metrics(feature_metrics, model_name, model_version, score, fecha_ejecucion)