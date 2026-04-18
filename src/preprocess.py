import pandas as pd


def _extract_title(name):
    if not isinstance(name, str):
        return "Unknown"

    if "," in name and "." in name:
        title = name.split(",", 1)[1].split(".", 1)[0].strip()
    else:
        title = "Unknown"

    # Group uncommon titles to keep categories stable.
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    return title_map.get(title, title)


def clean_data(df, use_feature_engineering=False):
    df = df.copy()

    if use_feature_engineering:
        if {"SibSp", "Parch"}.issubset(df.columns):
            df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
            df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

        if "Name" in df.columns:
            df["Title"] = df["Name"].apply(_extract_title)

    # Drop columns that are either high-cardinality text or not useful for training.
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    # Fill missing numeric and categorical values.
    df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    # Convert categoricals into numeric indicator columns for scikit-learn.
    categorical_columns = [column for column in ["Sex", "Embarked", "Title"] if column in df.columns]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

    return df
