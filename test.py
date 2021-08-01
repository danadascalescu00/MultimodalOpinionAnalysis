# from helpers import *

# base_dir = 'C:/Users/Dana/Desktop/Licenta/'
# datasets_names = ['MVSA_Single', 'MVSA_Multi']
# datasets_path = []
# for name in datasets_names:
#   datasets_path.append(os.path.join(base_dir, name + '/'))

# dir_save_files = os.path.join(base_dir, 'OpinionMining/Datasets')

# output_train_file_name = 'train.txt'
# output_validation_file_name = 'validation.txt'
# output_test_file_name = 'test.txt'

# # create a directory for the processed of the original MVSA dataset
# try:
#   if not os.path.exists(dir_save_files):
#     os.makedirs(dir_save_files) 
#     for dname in datasets_names:
#       os.makedirs(os.path.join(dir_save_files, dname))
      
#     print('directory created: {}'.format(dir_save_files))
#   else:
#     print('directory {} exists'.format(dir_save_files))
# except:
#   print('Unexpected error: ', sys.exc_info()[0])


# labels_file_name = 'labels.txt'
# text_filenames, image_filenames  = [], []
# # load raw data into variables
# labels_MVSA_Single, labels_MVSA_Multi = None, None
# text_MVSA_Single, text_MVSA_Multi = None, None

# for idx, dname in enumerate(datasets_names):
#   # change the working directory to the directory where we have all the date files of the current dataset
#   os.chdir(datasets_path[idx])
#   labels_MVSA_Single = pd.read_csv(labels_file_name, sep = "\t").dropna()
#   os.chdir(datasets_path[idx] + "/data/text")
#   text_filenames.append([file for file in glob.glob("*.txt")])
#   text_filenames[idx] = sorted(text_filenames[idx], key = get_file_index)

#   df = [pd.read_csv(file, header=None, sep=None, engine='python', usecols=None,squeeze=None) for file in text_filenames[idx]]
#   if idx == 0:
#     text_MVSA_Single = rearrange_dataframe(df)
#   else:
#     text_MVSA_Multi = rearrange_dataframe(df)

# # eliminate all the inconsistent data, i.e the posts where one of the labels is positive and the other is negative
print("\ud83c\udfca\ud83c\udfff")