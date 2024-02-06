#!/usr/bin/python 3
import argparse
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

""""The learntLoop class is a tool for training and evaluating Support Vector Machine (SVM) classifiers on tRNA 
sequence data obtained from Nanopore reads. It takes as input training and testing arrays in the form of .tsv or .csv 
files. The class performs one-hot encoding on the input data, shuffles and concatenates the training data, and then 
trains an SVM classifier. The trained classifier is used to predict the labels of testing data, and the results are 
analyzed, including accuracy, confusion matrix, and classification report."""


class learntLoop:
    def __init__(self, training_files, testing_files, label, spike_in, read_titrant, titrant_label, classification, query_reads):
        self.training_files = training_files
        self.testing_files = testing_files
        self.label = label
        self.spike_in = spike_in
        self.read_titrant = read_titrant
        self.titrant_label = titrant_label
        self.classification = classification
        self.query_reads = query_reads
        self.cnd_index = {}

        self.training_dfs = []
        self.testing_dfs = []
        self.combo_train = None
        self.test_df = None

        self.train_and_predict()

    '''This function takes a DataFrame (df) as input and performs one-hot encoding on its columns. It creates new 
    columns for each position and nucleotide, with column names like "posX_Y" where X is the position and Y is the 
    nucleotide (1 to 4). The resulting one-hot encoded DataFrame is returned.'''

    def oneHot(self, df):
        new_columns = {}
        for col in df.columns:
            for i in range(1, 5):
                new_col_name = f"pos{col.split('_')[1]}_{i}"
                new_columns[new_col_name] = df[col].astype(str).str.contains(str(i)).astype(bool)

        new_df = pd.DataFrame(new_columns)
        new_df = new_df[new_columns.keys()]
        return new_df

    '''This function takes a file path (condition) and a classifier label (classifier). It reads the data from the 
    specified file using pd.read_csv, performs one-hot encoding using the one_hot method, and then adds a 'class' 
    column to the DataFrame with the specified classifier label. The resulting DataFrame is returned.'''

    def get_dataframe(self, condition, classifier):
        df = self.oneHot(pd.read_csv(condition, sep='\t', index_col=0))
        df['class'] = classifier  # label the data as ctl or cnd
        return df

    '''This function takes a list of dataframes (df_list), cuts them down to the minimum read depth, concatenates them 
    into a single dataframe (combo_data), shuffles the concatenated dataframe, and returns the shuffled dataframe along 
    with the minimum read length.'''

    def shuffle_dataframe(self, df_list):
        min_length = min(len(df) for df in df_list)
        for i, df in enumerate(df_list):
            df_list[i] = df.iloc[:min_length]

        combo_training = pd.concat(df_list)  # concatenate training data
        shuffled_data = combo_training.sample(frac=1, random_state=42)  # shuffle data
        return shuffled_data, min_length

    '''This function takes the predictions, minimum length, and conditions as arguments and prints out the count of 
    each condition in the predictions along with the percentage.'''

    def print_stats(self, prediction, min_len, conditions):
        pred = list(prediction)
        for i in range(0, len(conditions)):
            if i in pred:
                print(conditions[i] + ': ', str(pred.count(i)) + '/' + str(min_len) + ' =', pred.count(i) / min_len)

    '''This function first creates dataframes from the training and testing files using the get_dataframe method.
    Then, it shuffles and concatenates the training dataframes, storing the result in self.combo_train, and shuffles
    the testing dataframes, storing the result in self.test_df. The min_len_train and min_len_test are also calculated
    during the shuffle process.'''

    def format_dataframe(self):
        # Get dataframes from input files
        cnd = ['IVT', 'WT', 'pus4', 'gcd10', 'trm2']  # hardcoded condition names
        training_dfs = []

        for i, file in enumerate(self.training_files):
            training_dfs.append(self.get_dataframe(file, i))
            guts = file.split('/')
            for x in guts:
                if x in cnd:
                    self.cnd_index[i] = x

        testing_dfs = []
        cnd_test_index = {}

        for i, file in enumerate(self.testing_files):
            testing_dfs.append(self.get_dataframe(file, i))
            guts = file.split('/')
            for x in guts:
                if x in cnd:
                    cnd_test_index[i] = x

        # Shuffle and concatenate dataframes
        combo_train, min_len_train = self.shuffle_dataframe(training_dfs)
        self.combo_train = combo_train

        test_df, min_len_test = self.shuffle_dataframe(testing_dfs)
        self.test_df = test_df

    '''This function extracts features and labels from the training dataframe, trains a Support Vector Machine (SVM) 
    classifier, and then predicts using the testing dataframe. Finally, it prints the results including accuracy, 
    confusion matrix, and classification report. If the user wants to query individual reads, it provides information 
    based on the specified query.'''

    def train_and_predict(self):
        self.format_dataframe()

        # Extract features (X) and labels (Y) from the training dataframe
        X_train = self.combo_train.iloc[:, :-1]
        Y_train = self.combo_train['class']

        # Create an SVM classifier
        clf = svm.SVC(decision_function_shape=self.classification)

        # Train the SVM classifier
        clf.fit(X_train, Y_train)

        # Extract features from the testing dataframe
        X_test = self.test_df.iloc[:, :-1]

        # Predict using the trained classifier
        predictions = clf.predict(X_test)

        # Print results
        tRNA_file = self.testing_files[0].split('/')[-1].split('.')[0].split('-')[-1]
        if tRNA_file.isdigit():
            if self.testing_files.split('/')[-1].split('.')[0].split('-')[-3] == 'CAT':
                tRNA = self.testing_files.split('/')[-1].split('.')[0].split('-')[-4] + \
                       self.testing_files.split('/')[-1].split('.')[0].split('-')[-3]
            else:
                tRNA = self.testing_files.split('/')[-1].split('.')[0].split('-')[-3]
        else:
            tRNA = tRNA_file

        print('\n')
        print(tRNA)
        self.print_stats(predictions, len(self.test_df), self.cnd_index)

        print('\n')
        accuracy = accuracy_score(self.test_df['class'], predictions)
        print("Accuracy:", accuracy)
        confusion_mat = confusion_matrix(self.test_df['class'], predictions)
        print('Confusion Matrix:')
        print(confusion_mat)
        classification_rep = classification_report(self.test_df['class'], predictions)
        print("Classification Report:")
        print(classification_rep)

        # If the user wants to query how individual reads were classified
        if self.query_reads is not None:
            confusion_matrix_query = self.query_reads

            # Create a DataFrame to hold the predicted and true labels
            results_df = pd.DataFrame({'True': self.test_df['class'], 'Predicted': predictions})

            # True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
            tp_df = results_df[(results_df['True'] == 1) & (results_df['Predicted'] == 1)]
            fp_df = results_df[(results_df['True'] == 0) & (results_df['Predicted'] == 1)]
            fn_df = results_df[(results_df['True'] == 1) & (results_df['Predicted'] == 0)]
            tn_df = results_df[(results_df['True'] == 0) & (results_df['Predicted'] == 0)]

            pd.set_option('display.max_rows', None)
            if confusion_matrix_query == 'TP':
                print("True Positives (TP):")
                print(tp_df)
            elif confusion_matrix_query == 'FP':
                print("\nFalse Positives (FP):")
                print(fp_df)
            elif confusion_matrix_query == 'FN':
                print("\nFalse Negatives (FN):")
                print(fn_df)
            elif confusion_matrix_query == 'TN':
                print("\nTrue Negatives (TN):")
                print(tn_df)


def main():

    # Define and parse arguments
    parser = argparse.ArgumentParser(
        description='Learning tRNA loops at the sequence level with Nanopore reads. The input arrays should be '
                    '.tsv or .csv files generated with getArray.py. The output is a statistical read out of how '
                    'many reads were classified as each condition and an accuracy measure.')
    parser.add_argument('-T', '--Training', nargs='+', metavar='filepaths', required=True,
                        help='Input at least two training array files. Each file is assigned a label, index '
                             'starting at 0, depending on the order of your input.')
    parser.add_argument('-t', '--testing', nargs='+', metavar='filepaths', required=True,
                        help='Input at least two testing array files. Each file is assigned a label, '
                             'index starting at 0, depending on the order of your input.')
    parser.add_argument('-l', '--label', type=int, metavar='int', required=False,
                        help='Designate the training label associated with the testing array. It should match the '
                             'index label of the corresponding training array (see -T help).')
    parser.add_argument('-q', '--query-reads', type=str, metavar='str', required=False,
                        choices=['TP', 'FP', 'FN', 'TN'],
                        help='Option to print out True Positive, False Positive, False Negative, or True Negative '
                             'classified reads,')
    parser.add_argument('-s', '--spike-in', type=float, metavar='float', required=False,
                        help='Option to spike in a percentage of reads from another array.')
    parser.add_argument('-r', '--read-titrant', metavar='filepath', required=False,
                        help='Input the array file containing reads that will be titrated.')
    parser.add_argument('-tl', '--titrant-label', metavar='int', required=False,
                        help='Assign the titrated reads array a label.')
    parser.add_argument('-c', '--classification', type=str, metavar='str', required=False, default='ovr',
                        help='Option to use ovo or ovr classification (Default ovr).')
    args = parser.parse_args()

    learntLoop(args.Training, args.testing, args.label, args.spike_in, args.read_titrant, args.titrant_label,
               args.classification, args.query_reads)


if __name__ == "__main__":
    main()
