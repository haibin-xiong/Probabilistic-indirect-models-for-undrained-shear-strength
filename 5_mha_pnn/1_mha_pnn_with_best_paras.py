import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
tfd = tfp.distributions
tfb = tfp.bijectors
from sklearn.model_selection import StratifiedKFold
import random
import os.path
import pathlib

seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
features = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']

all_su_pre = pd.read_excel(folder.parent / '0_based_multinormal_model' /'su_pre.xlsx')
labels = all_su_pre.iloc[:, -1].to_numpy()
mask = (labels >= 1) & (labels <= 7)
y_labels = labels[mask]

ext_data_t = pd.read_excel(folder.parent / '1_extented_data' /'ext_data_t.xlsx', sheet_name=f'Sheet_1')
mice_data = pd.read_excel(folder.parent / '2_imputation' /'mice_data.xlsx', sheet_name=f'Sheet_1')
mf_data = pd.read_excel(folder.parent / '2_imputation' /'mf_data.xlsx', sheet_name=f'Sheet_1')
target = pd.read_excel(folder.parent / '1_extented_data' /'su_r.xlsx', sheet_name=f'Sheet_1')

ext_data_t.columns = features
mice_data.columns = features
mf_data.columns = features

target_t = target.drop(columns=['dummy']).copy()
target_t = np.log(target_t).dropna().squeeze()
non_nan_index = target.dropna().index
ext_data_t = ext_data_t.loc[non_nan_index]
mice_data_t = mice_data.loc[non_nan_index]
mf_data_t = mf_data.loc[non_nan_index]
y_labels = y_labels[non_nan_index]
best_params_df = pd.read_excel('mha_pnn_best_params.xlsx')

def calculate_pre_with_uncertainty(predictions):
    predicted_samples = predictions.sample(1000).numpy()
    predicted_samples = np.squeeze(predicted_samples, axis=-1)
    pre = np.percentile(predicted_samples, [2.5, 50, 97.5], axis=0).T
    return pre

def predict_with_best_params(data_t, target_t, labels, best_params):

    units = best_params['units']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    activation_function = best_params['activation_function']
    num_heads = best_params['num_heads']
    key_dim = best_params['key_dim']

    nll_list = []
    data_r_list = []
    pre_r_list = []
    label_list = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(data_t, labels)):
        X_train, X_val = data_t.iloc[train_idx], data_t.iloc[test_idx]
        y_train, y_val = target_t.iloc[train_idx], target_t.iloc[test_idx]
        label_test = labels[test_idx]
        inputs = tf.keras.Input(shape=(X_train.shape[1],))
        x = tf.keras.layers.Dense(units, activation=activation_function)(inputs)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
        x = tf.expand_dims(x, axis=1)
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(x,
                                                                                                                     x)
        x = tf.keras.layers.Add()([x, attn_output])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
        x = tf.keras.layers.Dense(2)(x)
        outputs = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=tf.math.softplus(t[..., 1:]) + 1e-6))(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        negloglik = lambda y, rv_y: -rv_y.log_prob(y)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)
        # print(y_train.to_numpy().shape)
        model.fit(X_train.to_numpy(), y_train.to_numpy(),  validation_data=(X_val, y_val), batch_size=64, epochs=2000, verbose=0)
        # print(X_test.to_numpy().shape)
        predictions_test = model(X_val.to_numpy())
        # print(predictions_test.shape)
        score = model.evaluate(X_val.to_numpy(), y_val.to_numpy(), verbose=0)
        predictions_test  = calculate_pre_with_uncertainty(predictions_test)
        # print(pre_test.shape)
        pre_r_list.append(predictions_test)
        nll_list.append(score)
        data_r_list.append(y_val)
        label_list.append(label_test)
    return {
        "data_r_list": data_r_list,
        "pre_r_list": pre_r_list,
        "nll_list": nll_list,
        "label_list": label_list
    }

def format_pre_r_list(pre_r_list, data_r_list, label_list):
    all_preds = []
    for i, (pre_r, data_r, label) in enumerate(zip(pre_r_list, data_r_list, label_list)):
        df = pd.DataFrame(np.exp(pre_r), columns=["lower", "median", "upper"])
        df["real value"] = np.exp(np.array(data_r).reshape(-1))
        df["label"] = np.array(label).reshape(-1)
        df["fold"] = i
        all_preds.append(df)
    return pd.concat(all_preds, ignore_index=True)

datasets = {
    'ext': (ext_data_t, target_t, y_labels),
    'mice': (mice_data_t, target_t, y_labels),
    'mf': (mf_data_t, target_t, y_labels)
}

results_nll = []

for name, (data, target, labels) in datasets.items():
    print(f"Processing dataset: {name}")

    best_params = best_params_df[best_params_df["dataset_names"] == name].iloc[0].to_dict()
    print(best_params)
    prediction_likelihood = predict_with_best_params(data, target, labels, best_params)
    predictions_df = format_pre_r_list(prediction_likelihood["pre_r_list"], prediction_likelihood["data_r_list"], prediction_likelihood["label_list"])
    with pd.ExcelWriter(f"{name}_predictions_MHA.xlsx") as writer:
        for fold in sorted(predictions_df["fold"].unique()):
            fold_df = predictions_df[predictions_df["fold"] == fold]
            fold_df.to_excel(writer, sheet_name=f"fold_{fold+1}", index=False)

    nll_list = {
        "dataset": name,
        "nll_list": prediction_likelihood["nll_list"]
    }
    results_nll.append(nll_list)

df_results_likelihoods = pd.DataFrame(results_nll)
df_results_likelihoods.to_excel("nll_results.xlsx", index=False)