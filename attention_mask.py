attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape
input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
features = last_hidden_states[0][:,0,:].numpy()
