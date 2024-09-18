import pandas as pd

class TitanicDataPreprocessor:
    @staticmethod
    def preprocess_data(df):

        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Cabin'].fillna('Unknown', inplace=True)

        df['Title'] = df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
        df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        df['Title'] = df['Title'].replace(
            ['Dr', 'Rev', 'Col', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Don', 'Sir', 'Capt'], 'Other')

        df['TicketPrefix'] = df['Ticket'].apply(lambda ticket: ticket.split(' ')[0] if not ticket.isdigit() else 'NoPrefix')

        df['Cabin'] = df['Cabin'].apply(lambda cabin: cabin[0] if cabin != 'Unknown' else 'U')

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'TicketPrefix', 'Cabin'], drop_first=True)

        expected_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 
                            'Sex_male', 'Embarked_Q', 'Embarked_S', 'Title_Miss', 'Title_Mrs', 
                            'Title_Other', 'TicketPrefix_NoPrefix', 'Cabin_U']

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]

        return df
