from prepare_datasets import *
from Parameters import *
from DataSet import *

if __name__ == "__main__":

    params: Parameters = Parameters()
    params.use_cuda = torch.cuda.is_available()

    random.seed(params.SEED)
    np.random.seed(params.SEED)
    torch.manual_seed(params.SEED)
    torch.cuda.manual_seed_all(params.SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if params.use_cuda else {}


    dataset: DataSet = DataSet(params.cwd)
 
    # Creating data loaders for the text and image input from the train, validation and test sets
    text_train_dataset   = TensorDataset(torch.from_numpy(dataset.text_train), torch.from_numpy(dataset.text_train_labels))
    images_train_dataset = TensorDataset(torch.from_numpy(dataset.images_train), torch.from_numpy(dataset.images_train))

    text_val_dataset   = TensorDataset(torch.from_numpy(dataset.text_validation), torch.from_numpy(dataset.text_val_labels))
    images_val_dataset = TensorDataset(torch.from_numpy(dataset.images_validation), torch.from_numpy(dataset.image_validation_labels))
    
    text_test_dataset   = TensorDataset(torch.from_numpy(dataset.text_test), torch.from_numpy(dataset.text_test_labels))
    images_test_dataset = TensorDataset(torch.from_numpy(dataset.images_test), torch.from_numpy)

    text_train_loader  = DataLoader(text_train_dataset, batch_size = params.batch_size)
    image_train_loader = DataLoader(images_val_dataset, batch_size = params.batch_size)

    text_val_loader  = DataLoader(text_val_dataset, batch_size = params.batch_size)
    image_val_loader = DataLoader(images_train_dataset, batch_size=params.batch_size)

    text_test_loader  = DataLoader(text_test_dataset, batch_size = params.batch_size)
    image_test_loader = DataLoader(images_test_dataset, batch_size = params.batch_size)

    print(len(text_train_loader), len(image_train_loader))
    print(len(text_val_loader), len(image_val_loader))
    print(len(text_test_loader), len(image_test_loader))


    # save the params object
    # params_file_name = os.path.join(params.dir_save_files, 'params')
    # pickle.dump(params, open(params_file_name, 'wb'))