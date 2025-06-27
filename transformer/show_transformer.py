from cent.model.transformer import Encoder, Decoder, Transformer
import torch 
import torch.nn as nn

if __name__ == "__main__":
    # constants
    batch_size = 16                 # batch size
    max_feat_len = 100              # the maximum length of input sequence
    fbank_dim = 80                  # the dimension of input feature
    hidden_dim = 512                # the dimension of hidden layer
    vocab_size = 26                 # the size of vocabulary
    max_lable_len = 100             # the maximum length of output sequence

    # dummy data
    fbank_feature = torch.randn(batch_size, max_feat_len, fbank_dim)        # input sequence
    feat_lens = torch.randint(1, max_feat_len, (batch_size,))               # the length of each input sequence in the batch
    labels = torch.randint(0, 26, (batch_size, max_lable_len))              # output sequence
    label_lens = torch.randint(1, 10, (batch_size,))                        # the length of each output sequence in the batch

    # model
    feature_extractor = nn.Linear(fbank_dim, hidden_dim)      
                  # a single layer to simulate the audio feature extractor
    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, enc_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048
    )
    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, dec_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048, tgt_vocab_size=vocab_size
    )
    transformer = Transformer(feature_extractor, encoder, decoder, hidden_dim, vocab_size)

    # forward check
    logits = transformer(fbank_feature, feat_lens, labels)
    # print(logits.shape)     # (batch_size, max_label_len, vocab_size)
    assert logits.shape == (batch_size, max_lable_len, vocab_size)

    print(logits.shape) # 16, 100, 26
