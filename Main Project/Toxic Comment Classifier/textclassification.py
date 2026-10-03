
import json
import torch 
import nltk 
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def classify_text(text):
    embed_dim = 100
    hidden_size = 250
    num_layers=1
    output_size = 6 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    label_columns = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

    with open("vocab.txt", "r", encoding="utf-8") as file:
        vocab = json.load(file)


    user_input=text.lower()

    user_input_tokenized = nltk.word_tokenize(user_input)

    user_input_ids = [vocab.get(token, vocab["<UNK>"]) for token in user_input_tokenized]

    if not user_input_ids:
        raise ValueError("Please enter some text.")

    user_input_tensor = torch.tensor(user_input_ids,dtype=torch.long,device=device).unsqueeze(0)

    class LSTM(nn.Module):
        def __init__(self,vocab_size,embed_dim,hidden_size,num_layers,output_size,padding_id):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=padding_id) 
            self.lstm = nn.LSTM(input_size=embed_dim,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
            self.fc = nn.Linear(hidden_size*2, output_size) 
            self.padding_id = padding_id

        def forward(self, token_ids):
            lengths = (token_ids != self.padding_id).sum(dim=1)
            embedded = self.embedding(token_ids)

            packed = pack_padded_sequence(embedded, lengths.cpu(),batch_first=True,enforce_sorted=False)

            packed_output, (final_hidden, final_cell) = self.lstm(packed)

            # Convert the real LSTM outputs back into a batch tensor
            rnn_output, _ = pad_packed_sequence(packed_output,batch_first=True)

            # True for real words, False for PAD positions
            real_word_mask = (token_ids[:, :rnn_output.size(1)] != self.padding_id)

            # Prevent padding from being selected by max pooling
            rnn_output = rnn_output.masked_fill(~real_word_mask.unsqueeze(-1),float("-inf"))

            # Strongest feature found anywhere in each comment
            max_pooled = rnn_output.max(dim=1).values

            # Memory after the final real word
            last_hidden = final_hidden[-1]

            # Combine overall memory with strongest word-level evidence
            combined = torch.cat((last_hidden, max_pooled),dim=1)

            logits = self.fc(combined)

            return logits

    model = LSTM(vocab_size=len(vocab),embed_dim=embed_dim,hidden_size=hidden_size,num_layers=num_layers,output_size=output_size,padding_id=vocab["<PAD>"]).to(device)

    model.load_state_dict(torch.load("lstm_Final.pth", map_location=device, weights_only=True)) 
    model.to(device)
    model.eval()

    with torch.inference_mode():
        logits = model(user_input_tensor)
        probabilities = torch.sigmoid(logits)
        predictions = probabilities >= 0.5
    
    predicted_labels = [
        label
        for label, predicted in zip(
            label_columns, predictions[0].cpu().tolist()
        )
        if predicted
    ]

    scores = dict(zip(
        label_columns,
        probabilities[0].cpu().tolist()
    ))

    return predicted_labels, scores

