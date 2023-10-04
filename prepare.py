import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def create_index(df_one, date_col, datetime=True, index=True, sort=True):
    """ This function will re-index a DataFrame with the specified date column.
        df_one - DataFrame to alter.
        date_col - String name of the column with the date.
        datetime  - Will convert column to datetime.
        index - Will make the column the index.
        sort - Will sort the values so that they are in chronological order."""
    df = df_one.copy()
    if datetime:
        df[date_col] = pd.to_datetime(df[date_col])
    if index:
        df = df.set_index(date_col)
    if sort:
        df = df.sort_values(date_col)
    return df


def train_val_test(df, strat='None', seed=100, stratify=False):  # Splits dataframe into train, val, test
    """ This function will split my data into train, validate and test. It has the option to stratify."""
    if stratify:  # Will split with stratify if stratify is True
        train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=df[strat])
        val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=val_test[strat])
        print(train.shape, val.shape, test.shape)
        return train, val, test
    if not stratify:  # Will split without stratify if stratify is False
        train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
        val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
        print(f' train: {train.shape},  val: {val.shape},  test: {test.shape}')
        return train, val, test


def scale(train=None, val=None, test=None, dataframe=pd.DataFrame(), method='mms', scaled_cols=None, split=True):
    """This function will take in a dataframe or the train, val, test dataframes and scale the data according to
    whatever method is chosen."""
    df = dataframe.copy()
    if scaled_cols is None:
        scaled_cols = list(df.select_dtypes('number').columns)

    if split is not True:
        if method == 'mms':  # MinMax is chosen
            mms = MinMaxScaler()
            mms.fit(df[scaled_cols])
            df[scaled_cols] = mms.transform(df[scaled_cols])
            return df  # returns df
        if method == 'ss':  # Standard is chosen
            ss = StandardScaler()
            ss.fit(df[scaled_cols])
            df[scaled_cols] = ss.transform(df[scaled_cols])
            return df  # returns df
        if method == 'rs':  # Robust is chosen
            rs = RobustScaler()
            rs.fit(df[scaled_cols])
            df[scaled_cols] = rs.transform(df[scaled_cols])
            return df  # returns df

    if split is True:
        if train is None or val is None or test is None:
            train, val, test = train_val_test(df)
        if method == 'mms':  # MinMax is chosen
            mms = MinMaxScaler()
            mms.fit(train[scaled_cols])
            train[scaled_cols] = mms.transform(train[scaled_cols])
            val[scaled_cols] = mms.transform(val[scaled_cols])
            test[scaled_cols] = mms.transform(test[scaled_cols])
            return train, val, test
        if method == 'ss':  # Standard is chosen
            ss = StandardScaler()
            ss.fit(train[scaled_cols])
            train[scaled_cols] = ss.transform(train[scaled_cols])
            val[scaled_cols] = ss.transform(val[scaled_cols])
            test[scaled_cols] = ss.transform(test[scaled_cols])
            return train, val, test
        if method == 'rs':  # Robust is chosen
            rs = RobustScaler()
            rs.fit(train[scaled_cols])
            train[scaled_cols] = rs.transform(train[scaled_cols])
            val[scaled_cols] = rs.transform(val[scaled_cols])
            test[scaled_cols] = rs.transform(test[scaled_cols])
            return train, val, test  # returns train test and val


def dummies(train, val, test, drop_first=None, normal_list=None):
    """This function will one hot encode a dataframe. It accepts one or more dataframes, and two lists of columns."""
    if drop_first is not None:
        train = pd.get_dummies(train, columns=drop_first, drop_first=True)  # Drops first value from columns
        val = pd.get_dummies(val, columns=drop_first, drop_first=True)
        test = pd.get_dummies(test, columns=drop_first, drop_first=True)

    if normal_list is not None:
        train = pd.get_dummies(train, columns=normal_list)  # Does not drop first from this list of columns
        val = pd.get_dummies(val, columns=normal_list)
        test = pd.get_dummies(test, columns=normal_list)

    return train, val, test  # Returns encoded dataframes


def split_xy(df, target=''):
    """This function will split x and y according to the target variable."""
    x_df = df.drop(columns=target)
    y_df = df[target]
    return x_df, y_df  # Returns dataframe


def preprocess(dataframe):
    a = dataframe.copy()
    train, val, test = train_val_test(a, 'outcome_type', stratify=True)
    train, val, test = scale(train, val, test, scaled_cols=['age'])
    train, val, test = dummies(train, val, test, drop_first=['name', 'gender', 'neut_spay', 'condition'],
                               normal_list=['animal_type', 'breed1', 'breed2'])
    x_train, y_train = split_xy(train, 'outcome_type')  # split data to remove the target variable churn
    x_val, y_val = split_xy(val, 'outcome_type')
    x_test, y_test = split_xy(test, 'outcome_type')
    return train, val, test, x_train, y_train, x_val, y_val, x_test, y_test
