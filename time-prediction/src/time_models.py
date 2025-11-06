import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, timeseries_feat_dim, feat_dim, hidden=16, num_layer=2, out_feature=1):
        super(LSTM, self).__init__()
        self.in_feature_1 = timeseries_feat_dim
        self.in_feature_2 = feat_dim
        self.hidden = hidden
        self.num_layer = num_layer
        self.out_feature = out_feature

        # timeseries embedding layer
        self.rnncell = torch.nn.LSTM(self.in_feature_1, self.hidden, self.num_layer, bidirectional=True, batch_first=True)
        self.fc1 = torch.nn.Linear(self.hidden*2, self.hidden)
        
        # features embedding layer
        self.fc_embed_2_1 = nn.Linear(self.in_feature_2, self.hidden)
        self.fc_embed_2_2 = nn.Linear(self.hidden, self.hidden*2)
        self.fc_embed_2_3 = nn.Linear(self.hidden*2, self.hidden)
        self.fc_embed_2_ls = [self.fc_embed_2_1, self.fc_embed_2_2, self.fc_embed_2_3]

        # final layer
        self.fc_out_1 = nn.Linear(self.hidden*2, self.hidden)
        self.fc_out_2 = nn.Linear(self.hidden, self.out_feature)

        self.out_layer = nn.Linear(self.hidden, self.out_feature)
        
        for _, param in self.rnncell.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)
        
        self.activation = torch.nn.ReLU()

        return


    def forward(self, x_timeseries, x_features, feat_flag=False):
        x_timeseries = x_timeseries.permute(0, 2, 1)
        x_timeseries, h = self.rnncell(x_timeseries)
        out_timeseries = self.activation(self.fc1(x_timeseries))[:, -2:-1, :].squeeze()

        for l_features in self.fc_embed_2_ls:
            x_features = self.activation(l_features(x_features))
        out_features = x_features.squeeze()

        output = torch.cat((out_timeseries, out_features), dim=1)
        # return the feature level output
        if feat_flag:
            return output

        output = self.activation(self.fc_out_1(output))
        output = self.fc_out_2(output)

        return output



class GRU(nn.Module):
    def __init__(self, timeseries_feat_dim, feat_dim, hidden=16, num_layer=2, out_feature=1):
        super(GRU, self).__init__()
        self.in_feature_1 = timeseries_feat_dim
        self.in_feature_2 = feat_dim
        self.hidden = hidden
        self.num_layer = num_layer
        self.out_feature = out_feature

        # timeseries embedding layer
        self.rnncell = torch.nn.GRU(self.in_feature_1, self.hidden, num_layer, bidirectional=True, batch_first=True) # as bidirectional -> (hidden*2)*2
        self.fc1 = torch.nn.Linear(self.hidden*2, self.hidden)

        # features embedding layer
        self.fc_embed_2_1 = nn.Linear(self.in_feature_2, self.hidden)
        self.fc_embed_2_2 = nn.Linear(self.hidden, self.hidden*2)
        self.fc_embed_2_3 = nn.Linear(self.hidden*2, self.hidden)
        self.fc_embed_2_ls = [self.fc_embed_2_1, self.fc_embed_2_2, self.fc_embed_2_3]

        # final layer
        self.fc_out_1 = nn.Linear(self.hidden*2, self.hidden)
        self.fc_out_2 = nn.Linear(self.hidden, self.out_feature)
        
        for _, param in self.rnncell.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)
        
        self.activation = torch.nn.ReLU()
        
        return
    

    def forward(self, x_timeseries, x_features, feat_flag=False):
        x_timeseries = x_timeseries.permute(0, 2, 1)
        x_timeseries, h = self.rnncell(x_timeseries)
        out_timeseries = self.activation(self.fc1(x_timeseries))[:, -2:-1, :].squeeze()

        for l_features in self.fc_embed_2_ls:
            x_features = self.activation(l_features(x_features))
        out_features = x_features.squeeze()

        output = torch.cat((out_timeseries, out_features), dim=1)
        # return the feature level output
        if feat_flag:
            return output

        output = self.activation(self.fc_out_1(output))
        output = self.fc_out_2(output)
        
        return output


class Transformer(nn.Module):
    def __init__(self, timeseries_feat_dim, feat_dim, embedding_size=16, num_heads=4, num_encoder_layer=6, num_decoder_layer=6, out_feature=1):
        super(Transformer, self).__init__()
        self.in_feautre_1 = timeseries_feat_dim
        self.in_feature_2 = feat_dim
        self.embedding_size = embedding_size
        self.out_feature = out_feature

        # timeseries embedding layer
        self.fc_embed_1 = nn.Linear(timeseries_feat_dim, embedding_size)
        
        # features embedding layer
        self.fc_embed_2_1 = nn.Linear(self.in_feature_2, self.embedding_size)
        self.fc_embed_2_2 = nn.Linear(self.embedding_size, self.embedding_size*2)
        self.fc_embed_2_3 = nn.Linear(self.embedding_size*2, self.embedding_size)
        self.fc_embed_2_ls = [self.fc_embed_2_1, self.fc_embed_2_2, self.fc_embed_2_3]
        
        self.transformer = nn.Transformer(d_model=embedding_size, nhead=num_heads,num_encoder_layers=num_encoder_layer, num_decoder_layers=num_decoder_layer)
        
        self.fc_out_1 = nn.Linear(embedding_size*2, self.embedding_size)
        self.fc_out_2 = nn.Linear(embedding_size, self.out_feature)
        self.activation = nn.ReLU()

        return


    def forward(self, x_timeseries, x_features, feat_flag=False):
        x = x_timeseries.permute(2, 0, 1)
        x = self.activation(self.fc_embed_1(x))
        out_timeseries = self.transformer(x, x)[-1] # Encoder Transformer training : src, src
        
        for l_features in self.fc_embed_2_ls:
            x_features = self.activation(l_features(x_features))
        out_features = x_features.squeeze()

        out = torch.cat((out_timeseries, out_features), dim=1)
        # return the feature level output
        if feat_flag:
            return out

        out = self.activation(self.fc_out_1(out))
        out = self.fc_out_2(out)

        return out