from prepare_datasets import *
from Parameters import *
from DataSet import *
from SentimentClassifier import *

def run_text_sentiment_classifier(train_loader, validation_loader, test_loader, params, device):
    model = TextSentimentClassifier(freeze_bert=True).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    train_loss, validation_loss, test_loss = [], [], []
    validation_accuracy, test_accuracy = [], []
    average_recall, f1_score_pn = [], []


    def evaluate(model, dataloader):
        mean_acc, mean_loss = 0., 0.
        average_recall, f1_pn = 0., 0.
        count = 0

        model.eval()
        with torch.no_grad():
            for seq, attn_masks, labels in dataloader:
                logits = model(seq, attn_masks)
                mean_loss += F.cross_entropy(logits, labels.long()).item()
                mean_acc += get_accuracy_from_logits(logits, labels)
                tp, fn, fp = get_tp_fn_fp_from_logits(logits, labels)
                recall = get_recall(tp, fn)
                average_recall += get_avg_recall(recall)
                f1_pn += get_f1_pn(tp, fp, fn)

                count += 1

        return mean_acc / count, mean_loss / count, average_recall / count, f1_pn / count


    def train(model, optim, train_loader, val_loader, test_loader):
        best_accuracy = 0.

        scheduler = ReduceLROnPlateau(optim, 'min', patience = 1)
        for epoch in range(params.epochs):
            model.train()

            batch_losses = 0.
            count = 0

            for it, (sequences, attn_masks, labels) in enumerate(train_loader):
                # Clear gradients
                optim.zero_grad()
                # Obtaining the logits from the model
                logits = model(sequences, attn_masks)
                # Handling the unbalanced dataset
                class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels.numpy())
                class_weights = torch.tensor(class_weights, dtype = torch.float)
                # Computing loss
                loss = F.cross_entropy(logits, labels.long(), weight=class_weights)
                # Backpropagation
                loss.backward()
                # Optimization step
                optim.step()

                batch_losses += loss.item()
                count += 1

                if (it + 1) % params.print_every == 0:
                    acc = get_accuracy_from_logits(logits, labels)
                    print("    Iteration {} of epoch {} completed. Loss : {} Accuracy : {:.2f}".format(it+1, epoch+1, loss.item(), acc * 100.))

            epoch_loss = batch_losses / count
            train_loss.append(epoch_loss)

            val_acc, val_loss, _, _ = evaluate(model, val_loader)
            validation_loss.append(val_loss)
            validation_accuracy.append(val_acc)
            if val_acc > best_accuracy:
                print("Best validation accuracy improved from {} to {}, saving model...".format(best_accuracy, val_acc))
                best_accuracy = val_acc

            scheduler.step(val_loss)

            acc_test, loss_test, avg_recall, f1_pn = evaluate(model, test_loader)
            print("Epoch {} completed. Evaluation measurements on test dataset:\n\t Loss: {} \n\t Accuracy : {:.2f} \
                \n\t Average recall: {:.4f} \n\t F1_PN: {:.4f}\n".format(epoch, loss_test, acc_test * 100., avg_recall, f1_pn))
            test_accuracy.append(acc_test)
            test_loss.append(loss_test)
            average_recall.append(avg_recall)
            f1_score_pn.append(f1_pn)


    train(model, optimizer, train_loader, validation_loader, test_loader)

    plot_losses(train_loss, validation_loss, test_loss)


def run_sentiment_classifier(train_loader, validation_loader, test_loader, params, device):
    model = SentimentClassifier(freeze_bert=False).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    train_loss, validation_loss, test_loss = [], [], []
    validation_accuracy, test_accuracy = [], []
    average_recall, f1_score_pn = [], []

    def evaluate(model, dataloader):
        mean_acc, mean_loss = 0., 0.
        average_recall, f1_pn = 0., 0.
        count = 0

        model.eval()
        with torch.no_grad():
            for sequences, attn_masks, images, labels in dataloader:
                logits = model(sequences, attn_masks, images)
                mean_loss += F.cross_entropy(logits, labels.long()).item()
                mean_acc += get_accuracy_from_logits(logits, labels)
                tp, fn, fp = get_tp_fn_fp_from_logits(logits, labels)
                recall = get_recall(tp, fn)
                average_recall += get_avg_recall(recall)
                f1_pn += get_f1_pn(tp, fp, fn)

                count += 1

        return mean_acc / count, mean_loss / count, average_recall / count, f1_pn / count

    def train(train_loader, validation_loader, test_loader, params, device):
        logits = None
        best_accuracy = 0.
        best_test_accuracy = 0.
        best_model = None

        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1)
        for epoch in range(params.epochs):
            model.train()

            batch_losses = 0.
            count_iterations = 0

            for it, (sequences, attn_masks, images, labels) in enumerate(train_loader):
                sequences = sequences.to(device)
                attn_masks = attn_masks.to(device)
                images = images.to(device)

                # Clear gradients
                optimizer.zero_grad()
                # Forward pass: Obtaining the logits from the model
                logits = model(sequences, attn_masks, images)
                # Handling the unbalanced dataset
                class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels.numpy())
                class_weights = torch.tensor(class_weights, dtype = torch.float)
                # Computing loss
                loss = None
                if len(class_weights) == 2:
                    loss = F.cross(logits, labels.long())
                else:
                    loss = F.cross_entropy(logits, labels.long(), weight=class_weights)
                # Backpropagation pass
                loss.backward()
                # Optimization step
                optimizer.step()

                batch_losses += loss.item()
                count_iterations += 1

                if (it + 1) % params.print_every == 0:
                    acc = get_accuracy_from_logits(logits, labels)
                    print("    Iteration {} of epoch {} completed. Loss : {} Accuracy : {:.2f}".format(it+1, epoch+1, loss.item(), acc * 100.))
                
            epoch_loss = batch_losses / count_iterations
            train_loss.append(epoch_loss)

            val_acc, val_loss, _, _ = evaluate(model, validation_loader)
            validation_loss.append(val_loss)
            validation_accuracy.append(val_acc)
            if val_acc > best_accuracy:
                print("Best validation accuracy improved from {} to {}".format(best_accuracy, val_acc))
                best_accuracy = val_acc

            scheduler.step(val_loss)

            acc_test, loss_test, avg_recall, f1_pn = evaluate(model, test_loader)
            print("Epoch {} completed. Evaluation measurements on test dataset:\n\t Loss: {} \n\t Accuracy : {:.2f} \
                \n\t Average recall: {:.4f} \n\t F1_PN: {:.4f}\n".format(epoch+1, loss_test, acc_test * 100., avg_recall, f1_pn))
            
            if acc_test > best_test_accuracy:
                print("Best test accuracy improved from {} to {}, saving model...".format(best_test_accuracy, acc_test))
                best_test_accuracy = acc_test
                torch.save(model, os.path.join('C:\\Users\\Dana\\Desktop\\Licenta\\OpinionMining\\SaveFiles\\SentimentClassifier', 'model10_{:.0f}_{:.0f}'.format(acc_test * 100, f1_pn * 100 )))
            
            test_accuracy.append(acc_test)
            test_loss.append(loss_test)
            average_recall.append(avg_recall)
            f1_score_pn.append(f1_pn)
                

    train(train_loader, validation_loader, test_loader, params, device)

    plot_losses(train_loss, validation_loss, test_loss)


if __name__ == "__main__":

    params: Parameters = Parameters()
    params.use_cuda = torch.cuda.is_available()
    params.epochs = 40
    params.print_every = 20

    random.seed(params.SEED)
    np.random.seed(params.SEED)
    torch.manual_seed(params.SEED)
    torch.cuda.manual_seed_all(params.SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if params.use_cuda else {}

    dataset: DataSet = DataSet(params.cwd)

    # Creating data loaders for the text and image input from the train, validation and test sets
    multimodal_dataset_train: MultimodalDataset = MultimodalDataset(dataset.images_train, dataset.text_train, dataset.post_train_labels, 36)
    multimodal_dataset_val: MultimodalDataset = MultimodalDataset(dataset.images_validation, dataset.text_validation, dataset.post_val_labels, 36)
    multimodal_dataset_test: MultimodalDataset = MultimodalDataset(dataset.images_test, dataset.text_test, dataset.post_test_labels, 36)


    multimodal_train_loader = DataLoader(multimodal_dataset_train, batch_size = params.batch_size)
    multimodal_val_loader = DataLoader(multimodal_dataset_val, batch_size = params.batch_size)
    multimodal_test_loader = DataLoader(multimodal_dataset_test, batch_size = params.batch_size)

    # run_text_sentiment_classifier(text_train_loader, text_val_loader, text_test_loader, params, device)
    run_sentiment_classifier(multimodal_train_loader, multimodal_val_loader, multimodal_test_loader, params, device)