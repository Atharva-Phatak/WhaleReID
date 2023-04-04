from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def prepare_labels(y):
    # From here: https://www.kaggle.com/pestipeti/keras-cnn-starter
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    return integer_encoded, label_encoder


def prep_data(params):
    data = pd.read_csv(params.csv_path)
    new_whale_df = data[data.Id == "new_whale"]  # only new_whale dataset
    train_df = data[
        ~(data.Id == "new_whale")
    ]  # no new_whale dataset, used for training
    train_df["label"], label_encoder = prepare_labels(train_df.Id)
    # get validation set where we have one example of each class
    im_count = train_df.Id.value_counts()
    im_count.name = "sighting_count"
    train_df = train_df.join(im_count, on="Id")
    X_test = (
        train_df.sample(frac=1, random_state=params.seed)[
            (train_df.Id != "new_whale") & (train_df.sighting_count > 1)
        ]
        .groupby("label")
        .first()
    )
    X_test["label"] = pd.to_numeric(X_test.index)
    # Train on all images
    X_train = train_df
    # return {"train_data": X_train, "val_data" : X_test, "encoded_classes": label_encoder.classes_, "new_whale_data": new_whale_df}
    X_train.to_csv("./csv_store/train_data.csv", index=False)
    X_test.to_csv("./csv_store/val_data.csv", index=False)
    new_whale_df.to_csv("./csv_store/new_whale_data.csv", index=False)
    np.save("./csv_store/classes.npy", label_encoder.classes_)