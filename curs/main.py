import random

import pandas as pd
from sklearn.model_selection import train_test_split
from districts import *

import matplotlib.pyplot as plt


def map_locations(loc_dict):
    return {k: v for d in loc_dict for k, v in d.items()}


def shuffle_dataframe(df):
    shuffled_df = df.copy()
    for _ in range(1000):
        weights = [random.uniform(0, 1) for _ in range(len(shuffled_df))]

        total_weight = sum(weights)
        probabilities = [weight / total_weight for weight in weights]

        shuffled_indices = random.choices(range(len(shuffled_df)), weights=probabilities, k=len(shuffled_df))

        shuffled_df = df.iloc[shuffled_indices].reset_index(drop=True)

    return shuffled_df

def make_files_for_training(df: pd.DataFrame):
    X = df.drop('Price', axis=1)
    y = df['Price']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2)

    train_df = pd.DataFrame(X_train)
    train_df['Price'] = y_train.values

    val_df = pd.DataFrame(X_val)
    val_df['Price'] = y_val.values

    test_df = pd.DataFrame(X_test)
    test_df['Price'] = y_test.values

    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/validation.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)




def create_df():
    df = pd.read_csv('../data/csvdata.csv')

    df = df.drop('Unnamed: 0', axis=1)

    location_dict = map_locations([location_Delhi, location_Mumbai, location_Chennai, location_Hyderabad, location_Kolkata,
                                    location_Bangalore])

    city_to_number = {
        'Delhi': 1,
        'Mumbai': 2,
        'Bangalore': 3,
        'Kolkata': 4,
        'Chennai': 5,
        'Hyderabad': 6
    }


    df = df[df['Location'].isin(list(location_dict.keys()))]

    df['City'] = df['City'].map(city_to_number)
    df['Location'] = df['Location'].map(location_dict)


    df.to_csv('data/dataframe.csv', index=False)

    return df


def plot_locations(original_df, shuffled_df):
    plt.figure(figsize=(14, 6))

    # График для оригинального DataFrame
    plt.subplot(1, 2, 1)
    plt.scatter(original_df.index, original_df['Location'], alpha=0.7, color='blue')
    plt.title('Оригинальные данные (Location)')
    plt.xlabel('Индекс')
    plt.ylabel('Location')
    plt.xticks(rotation=90)

    # График для перемешанного DataFrame
    plt.subplot(1, 2, 2)
    plt.scatter(original_df.index, shuffled_df['Location'], alpha=0.7, color='orange')
    plt.title('Перемешанные данные (Location)')
    plt.xlabel('Индекс')
    plt.ylabel('Location')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig('1.png')



def main():
    df = create_df()


    shuffled_df = shuffle_dataframe(df)

    shuffled_df.to_csv('shdf.csv', index=False)

    # print(shuffled_df.shape[0])
    # print(shuffled_df.head())
    plot_locations(df, shuffled_df)


if __name__ == '__main__':
    main()



