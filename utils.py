from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np

# https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# random baseline: just based on the proportion that succeeed/fail pick a random value
class RandomModel():
    def __init__(self, p):
        """
        p: probability of predicting the positive class (1)
        """
        self.p = p
    
    def predict(self, x):
        return [np.random.choice([0,1], p=[1-self.p, self.p]) for _ in range(len(x))]

def load_data(data_root, settings, keywords):
    acquired, active, failed = pd.read_csv(f"{data_root}/acquired_companies.csv"), pd.read_csv(f"{data_root}/active_companies.csv"), pd.read_csv(f"{data_root}/failed_companies.csv")

    acquired = acquired[~acquired["longDescription"].str.contains("acquire", case=False, na=False)] # filter out any companies that mention "acquire" in their description to avoid data leakage
    print("acquired", acquired.shape, "active", active.shape, "failed", failed.shape)

    if settings.only_acquired:
        dataset = pd.concat([acquired, failed])
        print("dataset shape after only_acquired", dataset.shape)
    else:
        dataset = pd.concat([acquired, active, failed])

    dataset = dataset.dropna(subset=["tags"]) # get rid of null tags, yes we lose data :(


    if settings.filter_tech:
        # print("dataset shape after dropping null tags", dataset.shape)
        valid_ids = dataset["tags"].apply(lambda x: eval(x)).apply(lambda tags: any(tag in keywords for tag in tags))
        dataset = dataset[valid_ids]
        dataset = dataset.reset_index(drop=True)
        assert all(isinstance(x, str) for x in dataset["tags"].values)
    else:
        dataset = dataset.reset_index(drop=True)

    dataset = dataset.rename(columns={"tags": "features"})

    print("final dataset shape", dataset.shape)
    # print(dataset.head())
    print("number of failures in final dataset", (dataset["status"] == "Inactive").sum())

    original_indices = dataset.index

    # add other features here
    new_ds = dataset.copy()
    for idx in new_ds.index:
        row = new_ds.loc[idx]
        row_tags = eval(row["features"])
        if settings.include_country:
            row_country = str(row["country"])
            row_tags.append(row_country)
        new_ds.at[idx, "features"] = row_tags


    mlb = MultiLabelBinarizer()
    X_original = mlb.fit_transform(new_ds["features"])
    y_orig = np.array([0 if status=="Inactive" else 1 for status in new_ds["status"]])

    idx_to_class = {i: cls for i, cls in enumerate(mlb.classes_)}
    print("number of categories after mlb", len(idx_to_class))

    X_train_orig, X_test_orig, y_train_orig, y_test_orig, train_idx, test_idx = train_test_split(
        X_original, y_orig, original_indices, test_size=0.2, random_state=42, stratify=y_orig
    )

    # undersample ONLY for the train set since test set should be kept realistic
    if settings.undersample:
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train_orig, y_train_orig)
    else:
        X_train, y_train = X_train_orig, y_train_orig

    X_test, y_test = X_test_orig, y_test_orig

    return dataset, X_train, X_test, y_train, y_test, y_orig, train_idx, test_idx, idx_to_class

def eval_models(models, X_test, y_test, X_train, y_train, idx_to_class):
    perf_dict = {}
    for model in models:
        print(f"--- {model} ---")
        m = models[model]
        y_pred = m.predict(X_test)
        print("train f1 score", f1_score(y_train, m.predict(X_train)))
        print("val f1 score", f1_score(y_test, y_pred))
        print("val precision", precision_score(y_test, y_pred))
        print("val recall", recall_score(y_test, y_pred))

        perf_dict[model] = {
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        }
        

        if model == "Random Forest":
            print("top 10 feature importances for RF")
            importances = {idx_to_class[i]: importance for i, importance in enumerate(m.feature_importances_)}
            sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_importances[:10]:
                print(f"feature: {feature}, importance: {importance}")

    return perf_dict
