from helpers import *

class TextSentimentClassifier(nn.Module):
    def __init__(self, freeze_bert = True, num_classes = 3):
        super(TextSentimentClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.save_path = 'C:\\Users\\Dana\\Desktop\\Licenta\\OpinionMining\\SaveFiles\\TextSentimentClassifier\\'

        # Freeze layers
        if freeze_bert:
            for params in self.bert_model.parameters():
                params.requires_grad = False

        # Classification layer
        self.linear = nn.Linear(768, num_classes)

    def forward(self, sequences, attn_masks):
        # Feeding the input to BERT model in order to obtain the contextualized representations
        cont_reps = self.bert_model(sequences, attn_masks)
        cont_reps = cont_reps[0]

        cls_rep = cont_reps[:, 0]

        out = self.linear(cls_rep)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, d):
        super().__init__()

        W_w = nn.Parameter(torch.Tensor(d, d))
        nn.init.xavier_uniform_(W_w, gain=nn.init.calculate_gain('tanh'))
        W_os = nn.Parameter(torch.Tensor(d, d))
        nn.init.xavier_uniform_(W_os, gain=nn.init.calculate_gain('tanh'))
        bias = nn.Parameter(torch.Tensor(d))
        nn.init.zeros_(bias)
        u = nn.Parameter(torch.Tensor(d))
        nn.init.zeros_(u)

        self.W_w = W_w
        self.W_os = W_os
        self.bias = bias
        self.u = u

        self.tanh = torch.nn.Tanhshrink()
        # self.tanh = torch.nn.Tanh()

    def forward(self, visual_reps, contextual_text_reps):
        num_tokens = list(contextual_text_reps.size())[1]
        scene_object_reps = torch.repeat_interleave(visual_reps, repeats=num_tokens, dim=1)
        scene_object_reps = scene_object_reps.reshape(list(contextual_text_reps.size()))

        weighted_cont_reps = torch.matmul(contextual_text_reps, self.W_w)
        weighted_visual_reps = torch.matmul(scene_object_reps, self.W_os)

        u_t = self.tanh(weighted_cont_reps + weighted_visual_reps + self.bias)
        
        exp = torch.exp(torch.matmul(u_t, self.u))
        alpha = exp / (torch.sum(exp) + 1e-7) # Shape: [batch_size, num_tokens]


        weighted_text_reps = torch.matmul(alpha, contextual_text_reps) 
        v_alpha = torch.sum(weighted_text_reps, dim=1) # Shape: [batch_size, d]

        return v_alpha


class SentimentClassifier(nn.Module):
    def __init__(self, freeze_bert=False, num_classes=3):
        super(SentimentClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.vgg_model = models.vgg19()
        # self.vgg_model.eval()

        cwd = os.path.dirname(os.path.abspath(__file__))
        self.__scene_alexnet_file = os.path.join(cwd, 'alexnet_places365.pth.tar')
        model = models.__dict__['alexnet'](num_classes=365)
        checkpoint = torch.load(self.__scene_alexnet_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        # model.eval()
        self.alexnet_model = model

        # Freeze layers
        if freeze_bert:
            for params in self.bert_model.parameters():
                params.requires_grad = False

        self.linear1 = nn.Linear(in_features=1365, out_features=768)
        self.relu1 = nn.ReLU()
        self.attention_layer = AttentionLayer(768)
        self.linear2 = nn.Linear(in_features=1536, out_features=768)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=768, out_features=384)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(in_features=384, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.15)

        self.count_epochs = 1


    def forward(self, sequences, attn_masks, images, unfreeze_bert=True):
        if self.count_epochs >= 10:
            self.vgg_model.eval()
            self.alexnet_model.eval()

        if unfreeze_bert:
            for params in self.bert_model.parameters():
                params.requires_grad = True

        # Feeding the input to BERT model in order to obtain the contextualized representations
        cont_reps = self.bert_model(sequences, attn_masks)
        cont_reps = cont_reps[0]
        cls_rep = cont_reps[:, 0]
        d = list(cont_reps.size())[2]

        # Get the image features 
        # Object feature representation: Tensor of shape [batch_size, 1000], with confidence scores over Imagenet's 1000 classes. The output has unnormalized scores.
        with torch.no_grad():
            object_reps = self.vgg_model(images)

        # Scene feature representation: Tensor of shape [batch_size, 365], with confidence scores over Imagenet's 1000 classes. The output has unnormalized scores.
        with torch.no_grad():
            scene_reps = self.alexnet_model.forward(images)

        # Visual feature vector: Tensor of shape [batch_size, 1365].
        scene_object_reps = torch.cat((object_reps, scene_reps), dim=1)
        os = list(scene_object_reps.size())[1] # size of each visual feature vector

        visual_reps = self.relu1(self.linear1(scene_object_reps))

        attn_mask = self.attention_layer(visual_reps, cont_reps)
        attn = cls_rep * attn_mask

        multimodal_reps = torch.cat((visual_reps, attn), dim=1)
        V_mul = self.relu2(self.dropout(self.linear2(multimodal_reps)))
        V_int = self.relu3(self.dropout2(self.linear3(V_mul)))
        pred = self.softmax(self.linear4(V_int))

        self.count_epochs += 1

        return pred