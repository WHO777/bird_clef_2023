import pandas as pd


def filter_data(df, thresh=5):
    counts = df.primary_label.value_counts()
    cond = df.primary_label.isin(counts[counts < thresh].index.tolist())

    df['cv'] = True
    df.loc[cond, 'cv'] = False

    return df


def upsample_data(df, thresh=20, seed=42):
    class_dist = df['primary_label'].value_counts()

    down_classes = class_dist[class_dist < thresh].index.tolist()

    up_dfs = []

    for c in down_classes:
        class_df = df.query("primary_label==@c")
        num_up = thresh - class_df.shape[0]
        class_df = class_df.sample(n=num_up, replace=True, random_state=seed)
        up_dfs.append(class_df)

    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)

    return up_df
