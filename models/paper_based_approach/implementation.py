import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, BertModel
import numpy as np


class TextlessPhoneAligner(nn.Module):
    def __init__(self,
                 wav2vec_model='facebook/wav2vec2-base',
                 bert_model='bert-base-uncased',
                 frame_rate=10):
        super().__init__()

        # Speech Encoder (Wav2Vec2)
        self.speech_encoder = Wav2Vec2Model.from_pretrained(wav2vec_model)

        # Adjust stride for finer frame resolution
        if frame_rate == 10:
            self.speech_encoder.encoder.layers[0].attention.conv.stride = (1, 1)

        # Phone Encoder (Reduced BERT)
        self.phone_encoder = BertModel.from_pretrained(bert_model)
        self.phone_encoder.encoder.layer = self.phone_encoder.encoder.layer[:4]

        # Projection layers
        self.speech_proj = nn.Linear(768, 256)
        self.phone_proj = nn.Linear(768, 256)

        # Alignment parameters
        self.temperature = 0.1
        self.forward_sum_weight = 1.0

    def compute_similarity_matrix(self, speech_repr, phone_repr):
        # Project representations
        speech_proj = self.speech_proj(speech_repr)
        phone_proj = self.phone_proj(phone_repr)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(phone_proj, speech_proj.T)
        return similarity_matrix

    def forward_sum_loss(self, attention_matrix):
        # Implement forward sum loss to encourage monotonic alignment
        batch_size, phone_len, time_steps = attention_matrix.shape

        # Diagonal prior constraint
        diagonal_mask = torch.zeros_like(attention_matrix)
        for b in range(batch_size):
            for p in range(phone_len):
                start = max(0, p - 1)
                end = min(time_steps, p + 2)
                diagonal_mask[b, p, start:end] = 1

        constrained_attention = attention_matrix * diagonal_mask
        forward_sum_loss = -torch.sum(constrained_attention)

        return forward_sum_loss

    def contrastive_alignment_loss(self, hidden_states, quantized_embeddings):
        # Contrastive loss similar to Wav2Vec2 pretraining
        batch_size, time_steps, hidden_dim = hidden_states.shape

        # Sample negative examples
        negative_samples = torch.randint(0, len(quantized_embeddings),
                                         (batch_size, time_steps, 50))

        # Compute similarities
        positive_sim = F.cosine_similarity(hidden_states,
                                           quantized_embeddings.unsqueeze(0))
        negative_sims = F.cosine_similarity(
            hidden_states.unsqueeze(-1),
            quantized_embeddings[negative_samples]
        )

        # Contrastive loss
        loss = -torch.log(
            torch.exp(positive_sim / self.temperature) /
            (torch.exp(positive_sim / self.temperature) +
             torch.sum(torch.exp(negative_sims / self.temperature), dim=-1))
        )

        return loss.mean()

    def forward(self, audio_input, phone_input, quantized_embeddings):
        # Extract speech representations
        speech_repr = self.speech_encoder(audio_input).last_hidden_state

        # Extract phone representations
        phone_repr = self.phone_encoder(phone_input).last_hidden_state

        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(speech_repr, phone_repr)

        # Compute attention matrix
        attention_matrix = F.softmax(similarity_matrix, dim=0)

        # Compute losses
        contrastive_loss = self.contrastive_alignment_loss(
            speech_repr, quantized_embeddings)
        forward_sum_loss = self.forward_sum_loss(attention_matrix)

        # Combined loss
        total_loss = (contrastive_loss +
                      self.forward_sum_weight * forward_sum_loss)

        return total_loss, attention_matrix


# Curriculum learning training loop
def train_with_curriculum(model, datasets, optimizer):
    # Datasets: list of datasets with increasing complexity/length
    for dataset in datasets:
        for batch in dataset:
            audio_input = batch['audio']
            phone_input = batch['phones']
            quantized_embeddings = batch['quantized_embeddings']

            optimizer.zero_grad()
            loss, alignment = model(audio_input, phone_input, quantized_embeddings)
            loss.backward()
            optimizer.step()
