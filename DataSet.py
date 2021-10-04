from helpers import *

def overall_sentiment(row):
    # If the sentiment of one component is neutral and the other negative, the overall sentiment is negative
    if (row['text'] == 0 and row['image'] == 2) or (row['image'] == 2 and row['text'] == 0):
        return 0
    # If the sentiment of one component is positive and the other neutral, the overall sentiment is positive
    elif (row['text'] == 1 and row['image'] == 2) or (row['image'] == 2 and row['text'] == 1):
        return 1

    # Both components have the same sentiment
    return row['text']

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
        labels['overall_sentiment'] = labels.apply(lambda row: overall_sentiment(row), axis=1)

        # print('Distribuția claselor în %{self.dataset_name}:')
        # print('Text:\n', labels['text'].value_counts())
        # print('Images:\n', labels['image'].value_counts())
        # print('Posts:\n', labels['overall_sentiment'].value_counts())
        
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

        # Apply the padding strategy
        train_text_lengths = df_text_train['text'].apply(lambda x : len(x.split(' ')))
        val_text_lengths = df_text_val['text'].apply(lambda x : len(x.split(' ')))
        test_text_lengths = df_text_test['text'].apply(lambda x : len(x.split(' ')))

        self.max_text_length = int(max(np.mean(train_text_lengths), np.mean(val_text_lengths), \
            np.mean(test_text_lengths))) + 5

        # print(df_text_train.head())
        # print(df_text_val.head())
        # print(df_text_test.head())

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
        self.__y_images_test = y_test['image'].values

        # self.__y_post_train = self.get_labels(y_train['text'].values, y_train['image'].values)
        # self.__y_post_val = self.get_labels(y_val['text'].values, y_val['image'].values)
        # self.__y_post_test = self.get_labels(y_test['text'].values, y_test['image'].values)
        self.__y_post_train = y_train['overall_sentiment'].values
        self.__y_post_val = y_val['overall_sentiment'].values
        self.__y_post_test = y_test['overall_sentiment'].values

    def get_labels(self, y_text, y_images):
        y_text, y_images = np.array(y_text), np.array(y_images)
        y_post = [get_label(a, b) for a, b in zip(y_text, y_images)]
        y_post = np.array(y_post).astype('float32')

        return y_post

    @property
    def text_train(self):
        return np.array(self.__X_text_train, dtype = str)

    @property
    def text_validation(self):
        return np.array(self.__X_text_val, dtype = str)

    @property
    def text_test(self):
        return np.array(self.__X_text_test, dtype = str)

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

    @property
    def post_train_labels(self):
        return self.__y_post_train

    @property
    def post_val_labels(self):
        return self.__y_post_val

    @property
    def post_test_labels(self):
        return self.__y_post_test


class TextDataset(Dataset):
    def __init__(self, X, y, max_length, device):
        self.X = X
        self.y = y
        self.max_length = max_length
        self.device = device

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_punctuation_indexes(self, tokens):
        indexes = []
        for idx, token in enumerate(tokens[:-1]):
            if token in punctuation and not tokens[idx + 1] in punctuation:
                indexes.append(idx)

        return indexes

    def add_special_tokens(self, tokens, punctuation_indexes):
        for i, index in enumerate(punctuation_indexes):
            tokens = tokens[:index + i + 1] + ['[SEP]'] + tokens[index + i + 1:]

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        return tokens

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]

        # Pre-processing the data to be suitable for the BERT model
        tokens = self.tokenizer.tokenize(sample)
        indexes = self.get_punctuation_indexes(tokens)
        tokens = self.add_special_tokens(tokens, indexes)

        if len(tokens) < self.max_length:
            tokens = tokens + ['[PAD]' for _ in range(self.max_length - len(tokens))]
        else:
            tokens = tokens[:self.max_length-1] + ['[SEP]'] # Prunning the list to be of specified max length

        # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        tokens_ids = torch.tensor(tokens_ids, device=self.device) 

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids != 0).long()
        attn_mask = torch.tensor(attn_mask, device=self.device)

        label = torch.tensor(label, device=self.device)

        return tokens_ids, attn_mask, label


class MultimodalDataset(Dataset):
    def __init__(self, X_image, X_text, y, max_length):
        self.X = X_text
        self.X_image = X_image
        self.y = y
        self.max_length = max_length

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_punctuation_indexes(self, tokens):
        indexes = []
        for idx, token in enumerate(tokens[:-1]):
            if token in punctuation and not tokens[idx + 1] in punctuation:
                indexes.append(idx)

        return indexes

    def add_special_tokens(self, tokens, punctuation_indexes):
        for i, index in enumerate(punctuation_indexes):
            tokens = tokens[:index + i + 1] + ['[SEP]'] + tokens[index + i + 1:]

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        return tokens

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        image = self.X_image[idx]
        label = self.y[idx]

        # Pre-processing the data to be suitable for the BERT model
        tokens = self.tokenizer.tokenize(sample)
        indexes = self.get_punctuation_indexes(tokens)
        tokens = self.add_special_tokens(tokens, indexes)

        if len(tokens) < self.max_length:
            tokens = tokens + ['[PAD]' for _ in range(self.max_length - len(tokens))]
        else:
            tokens = tokens[:self.max_length-1] + ['[SEP]'] # Prunning the list to be of specified max length

        # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        tokens_ids = torch.tensor(tokens_ids) 

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids != 0).long()
        attn_mask = torch.tensor(attn_mask)

        label = torch.tensor(label)

        return tokens_ids, attn_mask, image, label
        