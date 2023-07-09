from os.path import exists, join
from os import mkdir
import json
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class RNNModel(nn.Module):
    def __init__(self, hyperparams, device):
        super(RNNModel, self).__init__()
        #Set cpu/gpu device
        self.device=device
        self.to(self.device)

        # Defining some parameters
        self.hyperparams=hyperparams
        self.direction="forward"
        self.hidden_dim = hyperparams["hidden_dim"]
        self.embedding_dim = hyperparams["embedding_dim"]
        self.n_layers = hyperparams["n_rnn_layers"]
        self.output_size = hyperparams["output_size"]
        

        #Defining the layers
        #Input
        self.embedding = nn.Embedding(self.output_size, self.embedding_dim)
        # RNN Layer
        self.num_directions= 2 if self.direction=="bidirectional" else 1
        #self.hidden_dim=[self.hidden_dim, self.hidden_dim*2][self.direction=="bidirectional"]
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=(self.direction == "bidirectional"))
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim*self.num_directions, self.output_size)

    def forward(self, x):

        #Send input to device (gpu/cpu)
        x.to(self.device)

        #Here you define the forward pass, calling the layers you have defined in the constructor (init function)
        batch_size = x.size(0)
        
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        #Embed the words
        embeded = self.embedding(x)

        # Passing in the input and hidden state into the model and obtaining outputs
        #x.shape: (batch, sentence_length)
        #(num_layers * num_directions, batch, hidden_size)
        #out shape: (seq_len, batch, num_directions * hidden_size)
        out, hidden = self.rnn(embeded, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        out = F.log_softmax(out, dim=2)

        return out, hidden
    
    def init_hidden(self, batch_size):
        # Generates the first hidden state (just zeros)
        # (num_layers * num_directions, batch, hidden_size)
        hidden = torch.zeros(self.n_layers*self.num_directions, batch_size, self.hidden_dim, device=self.device)
        return hidden
    
    def save_model(self, path_to_saved_models, mapper,  epochs_trained, args, additional_description=""):
        """save hyperparameters, weights, model and mappings. """

        modelfolder=join(path_to_saved_models, "model_%s_%idelay_seed_%i_epoch_%i/"%(args.language, args.delay, args.seed, epochs_trained))
        if not exists(modelfolder):
            mkdir(modelfolder)
        
        #Save hyperparameters
        json.dump(self.hyperparams, open( join(modelfolder, "hyperparams.json"), 'w' ) )
            
        #Save weights in a readable format, for further analyses
        for label, weights in self.state_dict().items():
            fname=join(modelfolder, "%s_weights.json"%label)
            if self.device.type == 'cuda':
                weights_in_mem=weights.cpu()
                npw=weights_in_mem.numpy()
            else:
                npw=weights.numpy()
            np.savetxt(fname, npw)
    
        #Save model
        torch.save(self.state_dict(), join(modelfolder, "model"))

        #Save mappings
        if mapper is not None:
            mapper.save(join(modelfolder, "w2i"))

        #Save file with command line arguments (containing hyperparams but also input file, seed, etc)
        json.dump(vars(args), open( join(modelfolder, "args.json"), 'w' ) )

        return modelfolder
