from helpers import *

class DataSet:
    def __init__(self, current_working_directory):
        self.dataset_name = 'MVSA_Single' # dataset name: MVSA_Single/MVSA_Multi
        self.dataset_dir = 'Datasets/%s' %self.dataset_name
        self.cwd = current_working_directory
        self.texts_path = join_path(self.cwd, self.dataset_dir, 'text')
        self.images_path = join_path(self.cwd, self.dataset_dir, 'image')

        os.chdir(self.dataset_dir)

        labels = pd.read_csv('labels.csv', sep='\t').dropna()
        labels = labels.sort_values(by = ['ID'])
        labels = labels.replace(sentiment_label)
        
        # load raw data into variable from the text and image files
        os.chdir(self.texts_path)
        text_filenames = [file for file in glob.glob("*.txt")]

        os.chdir(self.images_path)
        image_files = [file for file in glob.glob("*.jpg")]

        # Split the text set in train, validation and test sets with 8:1:1 ratio
        text_files_train, text_files_test, image_files_train, image_files_test = \
            train_test_split(text_filenames, image_files, test_size = 0.1)
        text_files_train, text_files_val, image_files_train, image_files_val = \
            train_test_split(text_files_train, image_files_train, test_size = 0.1)

        text_files_train, image_files_train = sorted(text_files_train, key = get_file_index), sorted(image_files_train, key = get_file_index)
        text_files_val, image_files_val = sorted(text_files_val, key = get_file_index), sorted(image_files_val, key = get_file_index)
        text_files_test, image_files_test = sorted(text_files_test, key = get_file_index), sorted(image_files_test, key = get_file_index)

        idx_train = list(map(get_file_index, text_files_train))
        idx_val = list(map(get_file_index, text_files_val))
        idx_test = list(map(get_file_index, text_files_test))

        df_text_train = get_text_dataframe(self.texts_path, text_files_train)
        df_text_val = get_text_dataframe(self.texts_path, text_files_val)
        df_text_test = get_text_dataframe(self.texts_path, text_files_test)

        """
            Pre-processing the text data
        """
        # STEP 1: Convert emojis to their corresponding words
        df_text_train['text'] = df_text_train['text'].apply(demojize)
        df_text_val['text'] = df_text_val['text'].apply(demojize)
        df_text_test['text'] = df_text_test['text'].apply(demojize)
        # STEP 2: Convert emoticons to their corresponding words
        df_text_train['text'] = df_text_train['text'].apply(replace_emoticons)
        df_text_val['text'] = df_text_val['text'].apply(replace_emoticons)
        df_text_test['text'] = df_text_test['text'].apply(replace_emoticons)
        # STEP 3: Decode text in "UTF-8" and normalize it
        df_text_train['text'] = df_text_train['text'].apply(lambda x : normalize_text(x))
        df_text_val['text'] = df_text_val['text'].apply(lambda x : normalize_text(x))
        df_text_test['text'] = df_text_test['text'].apply(lambda x : normalize_text(x))
        # STEP 4: Replace URLs with the <URL> tag
        df_text_train['text'] = df_text_train['text'].apply(lambda x : re.sub(url_addresses_reg, ' <URL> ', x))
        df_text_val['text'] = df_text_val['text'].apply(lambda x : re.sub(url_addresses_reg, ' <URL> ', x))
        df_text_test['text'] = df_text_test['text'].apply(lambda x : re.sub(url_addresses_reg, ' <URL> ', x))
        # STEP 5: Remove all the email addresses
        df_text_train['text'] = df_text_train['text'].apply(lambda x : re.sub(mail_reg, ' ', x))
        df_text_val['text'] = df_text_val['text'].apply(lambda x : re.sub(mail_reg, ' ', x))
        df_text_test['text'] = df_text_test['text'].apply(lambda x : re.sub(mail_reg, ' ', x))
        # STEP 6: Split attached words
        df_text_train['text'] = df_text_train['text'].apply(split_attached_words)
        df_text_val['text'] = df_text_val['text'].apply(split_attached_words)
        df_text_test['text'] = df_text_test['text'].apply(split_attached_words)
        # STEP 7: Lower casing all tweets
        df_text_train['text'] = df_text_train['text'].apply(lambda x : x.lower())
        df_text_val['text'] = df_text_val['text'].apply(lambda x : x.lower())
        df_text_test['text'] = df_text_test['text'].apply(lambda x : x.lower())
        # STEP 8: Replace any sequence of the same letter of length greater than 2 with a sequence of length 2
        df_text_train['text'] = df_text_train['text'].apply(remove_multiple_occurences)
        df_text_val['text'] = df_text_val['text'].apply(remove_multiple_occurences)
        df_text_test['text'] = df_text_test['text'].apply(remove_multiple_occurences)
        # STEP 9: Abbreviated words are extended to the form in the dictionary
        df_text_train['text'] = df_text_train['text'].apply(expand_words)
        df_text_val['text'] = df_text_val['text'].apply(expand_words)
        df_text_test['text'] = df_text_test['text'].apply(expand_words)
        # STEP 10: Eliminate numerical and special characters 
        df_text_train['text'] = df_text_train['text'].apply(lambda x : x.translate(str.maketrans(' ', ' ', special_characters)))
        df_text_val['text'] = df_text_val['text'].apply(lambda x : x.translate(str.maketrans(' ', ' ', special_characters)))
        df_text_test['text'] = df_text_test['text'].apply(lambda x : x.translate(str.maketrans(' ', ' ', special_characters)))
        # # STEP 11(Optional): Remove the old style retweet text
        df_text_train['text'] = df_text_train['text'].apply(lambda x : re.sub(retweets_reg, ' ', x))
        df_text_val['text'] = df_text_val['text'].apply(lambda x : re.sub(retweets_reg, ' ', x))
        df_text_test['text'] = df_text_test['text'].apply(lambda x : re.sub(retweets_reg, ' ', x))

        print(df_text_train.head())
        print(df_text_val.head())
        print(df_text_test.head())

        images_train = get_images(self.images_path, image_files_train)
        images_val = get_images(self.images_path, image_files_val)
        images_test = get_images(self.images_path, image_files_test)

        y_train = labels[labels['ID'].isin(idx_train)]
        y_train = y_train.sort_values(by = ['ID'])
        del(idx_train)

        y_val = labels[labels['ID'].isin(idx_val)]
        y_val = y_val.sort_values(by = ['ID'])
        del(idx_val)

        y_test = labels[labels['ID'].isin(idx_test)]
        y_test = y_test.sort_values(by = ['ID'])
        del(idx_test)


        self.__X_text_train = df_text_train['text'].values
        self.__X_text_val = df_text_val['text'].values
        self.__X_text_test = df_text_test['text'].values
        self.__y_text_train = y_train['text'].values
        self.__y_text_val = y_val['text'].values
        self.__y_text_test = y_test['text'].values
        self.__X_images_train = images_train
        self.__X_images_val = images_val
        self.__X_images_test = images_test
        self.__y_images_train = y_train['image'].values
        self.__y_images_val = y_val['image'].values
        self.__y_images_test = y_val['image'].values


    @property
    def text_train(self):
        return np.array(self.__X_text_train, dtype = str)

    @property
    def text_validation(self):
        return np.array(self.__X_text_train, dtype = str)

    @property
    def text_test(self):
        return np.array(self.__X_text_train, dtype = str)

    @property
    def images_train(self):
        return self.__X_images_train

    @property
    def images_validation(self):
        return self.__X_images_val

    @property
    def images_test(self):
        return self.__X_images_test

    @property
    def text_train_labels(self):
        return np.array(self.__y_text_train).astype('float32')

    @property
    def text_validation_labels(self):
        return np.array(self.__y_text_val).astype('float32')

    @property
    def text_test_labels(self):
        return np.array(self.__y_text_test).astype('float32')

    @property
    def image_train_labels(self):
        return np.array(self.__y_images_train).astype('float32')

    @property
    def image_validation_labels(self):
        return np.array(self.__y_images_val).astype('float32')

    @property
    def image_test_labels(self):
        return np.array(self.__y_images_test).astype('float32')