from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np

#need to pip install transformers torch pandas numpy


def generateEmbeddings(inputCsv, outputCsv):
    # Load data 
    df = pd.read_csv(f"../data/{inputCsv}")

    # Load Bert
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")


    #Convert notes into embedddings/vectors
    embeddings = []
    for desc, label in zip(df['description'], df['label_id']):
        inputs = tokenizer(desc, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:,0,:].squeeze().numpy()
        # Combine label and embedding into a single array
        row = np.concatenate(([label], cls_embedding))
        embeddings.append(row)
        
    np.savetxt(f"../data/{outputCsv}", embeddings, delimiter=",")
    
generateEmbeddings("samples_clean.csv", "cleanEmbeddings.csv")
generateEmbeddings("samples_noisy.csv", "noisyEmbeddings.csv")
