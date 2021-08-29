from helpers import *

datasets_path = []
for name in datasets_names:
    datasets_path.append(os.path.join(base_dir, name + '/'))

dir_save_files = os.path.join(base_dir, 'OpinionMining/Datasets')

text_extension_file = ".txt"
labels_filename = 'labels.txt'

if not os.path.exists(dir_save_files):
# create a directory for the processed files of the original MVSA dataset
    try:
        os.makedirs(dir_save_files)
        for dname in datasets_names:
            dataset_path = os.path.join(dir_save_files, dname)
            os.makedirs(dataset_path)
            os.makedirs(os.path.join(dataset_path, "text"))
            os.makedirs(os.path.join(dataset_path, "image"))
        print('directory created: {}'.format(dir_save_files))
    except:
        print('Unexpected error: ', sys.exc_info()[0])

    text_filenames, image_filenames = [], []
    labels_MVSA_Single, labels_MVSA_Multi = None, None

    for idx, dataset in enumerate(datasets_names):
        # change the working directory to the directory where we have all the data files of the current dataset
        os.chdir(datasets_path[idx])
        if dataset == 'MVSA_Single':
            labels_MVSA_Single = pd.read_csv(labels_filename, sep = "\t").dropna()
        else:
            labels_MVSA_Multi = pd.read_csv(labels_filename, sep = "\t").dropna()

        os.chdir(datasets_path[idx] + "/data/text")
        text_filenames.append([file for file in glob.glob("*.txt")])
        os.chdir(datasets_path[idx] + "/data/images")
        image_filenames.append([file for file in glob.glob("*.jpg")])

    for column in labels_MVSA_Single.columns[1:]:
        labels_MVSA_Single[column] = labels_MVSA_Single[column].apply(lambda y : sentiment_label[y])

    for column in labels_MVSA_Multi.columns[1:]:
        labels_MVSA_Multi[column] = labels_MVSA_Multi[column].apply(lambda y : sentiment_label[y])


    # the final label of the samples in MVSA_Multi is represented by the majority vote of the annotators
    indexes = labels_MVSA_Multi[~((labels_MVSA_Multi['text1'] == labels_MVSA_Multi['text2']) 
        | (labels_MVSA_Multi['text2'] == labels_MVSA_Multi['text3']) | (labels_MVSA_Multi['text1'] == labels_MVSA_Multi['text3']))
        | ~((labels_MVSA_Multi['image1'] == labels_MVSA_Multi['image2']) | (labels_MVSA_Multi['image2'] == labels_MVSA_Multi['image3']) 
        | (labels_MVSA_Multi['image1'] == labels_MVSA_Multi['image3']))].index

    labels_MVSA_Multi.drop(indexes, inplace=True)

    labels_MVSA_Multi['text'] = labels_MVSA_Multi.filter(like = 'text').mode(axis = 1).iloc[:,0]
    labels_MVSA_Multi['image'] = labels_MVSA_Multi.filter(like = 'image').mode(axis = 1).iloc[:,0]

    labels_MVSA_Multi = labels_MVSA_Multi.drop(columns = ["text1", "text2", "text3", "image1", "image2", "image3"])

    # eliminate all the inconsistent data, i.e the posts where one of the labels is positive and the other is negative
    indexes = labels_MVSA_Single[((labels_MVSA_Single['text'] == 0) & (labels_MVSA_Single['image'] == 1)) \
        | ((labels_MVSA_Single['text'] == 1) & (labels_MVSA_Single['image'] == 0))].index
    labels_MVSA_Single.drop(indexes, inplace=True)
    labels_MVSA_Single.to_csv(base_dir + 'MVSA_Single/labels.csv', sep = '\t', index = False)
    print('Number of valid Tweets in MVSA-Single: ', len(labels_MVSA_Single.index))

    indexes = labels_MVSA_Multi[((labels_MVSA_Multi['text'] == 0) & (labels_MVSA_Multi['image'] == 1)) \
        | ((labels_MVSA_Multi['text'] == 1) & (labels_MVSA_Multi['image'] == 0))].index
    labels_MVSA_Multi.drop(indexes, inplace=True)
    labels_MVSA_Multi.to_csv(base_dir + 'MVSA_Multi/labels.csv', sep = '\t', index = False)
    print('Number of valid Tweets in MVSA_Multi: ', len(labels_MVSA_Multi.index))

    # save the new datasets
    files_MVSA = [labels_MVSA_Single['ID'].astype(str).to_list() , labels_MVSA_Multi['ID'].astype(str).to_list()]
    for idx, dataset in enumerate(datasets_names):
        cwd = base_dir + dataset
        # move all the text files to the destination directory
        move_files(files_MVSA[idx], '.txt', cwd + '/data/text', dir_save_files + '/' + dataset + '/text')
        # move all the image files to the destination directory
        move_files(files_MVSA[idx], '.jpg', cwd + '/data/images', dir_save_files + '/' + dataset + '/image')

else:
    print('directory {} exists'.format(dir_save_files))