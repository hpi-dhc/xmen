import torch
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
)

# Adapted from https://github.com/cambridgeltl/sapbert/blob/main/src/model_wrapper.py

_EMBED_DIM = 768


class Model_Wrapper(object):
    """
    Wrapper class for BERT encoder
    """

    def __init__(self):
        self.tokenizer = None
        self.encoder = None

    def get_dense_encoder(self):
        assert self.encoder is not None
        return self.encoder

    def get_dense_tokenizer(self):
        assert self.tokenizer is not None

        return self.tokenizer

    def save_model(self, path, context=False):
        # save bert model, bert config
        self.encoder.save_pretrained(path)

        # save bert vocab
        self.tokenizer.save_pretrained(path)

    def load_model(self, path, max_length=25, use_cuda=True, lowercase=True):
        self.load_bert(path, max_length, use_cuda)

        return self

    def load_bert(self, path, max_length, use_cuda, lowercase=True):
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, do_lower_case=lowercase)
        self.encoder = AutoModel.from_pretrained(path)
        if use_cuda:
            self.encoder = self.encoder.cuda()

        return self.encoder, self.tokenizer

    def embed_dense(
        self, names, show_progress=False, use_cuda=True, batch_size=2048, agg_mode="cls", memory_map_file=None
    ):
        """
        Embedding data into dense representations

        Parameters
        ----------
        names : np.array
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        self.encoder.eval()  # prevent dropout

        shape = (len(names), _EMBED_DIM)
        batch_size = batch_size
        if memory_map_file:
            dense_embeds = np.memmap(memory_map_file, dtype="float32", mode="w+", shape=shape)
        else:
            dense_embeds = np.zeros(shape, dtype=np.float32)

        with torch.no_grad():
            if show_progress:
                iterations = tqdm(range(0, len(names), batch_size))
            else:
                iterations = range(0, len(names), batch_size)

            for start in iterations:
                end = min(start + batch_size, len(names))
                batch = names[start:end]
                batch_tokenized_names = self.tokenizer.batch_encode_plus(
                    batch,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=25,
                    padding="max_length",
                    return_tensors="pt",
                )

                # generate cuda names only if use cuda
                # and overwrite original batch_tokenized_names
                if use_cuda:
                    batch_tokenized_names_cuda = {}
                    for k, v in batch_tokenized_names.items():
                        batch_tokenized_names_cuda[k] = v.cuda()
                    batch_tokenized_names = batch_tokenized_names_cuda

                # batch_tokenized_names is passed (-_cuda has been removed)
                last_hidden_state = self.encoder(**batch_tokenized_names)[0]
                if agg_mode == "cls":
                    batch_dense_embeds = last_hidden_state[:, 0, :]  # [CLS]
                elif agg_mode == "mean_all_tok":
                    batch_dense_embeds = last_hidden_state.mean(1)  # pooling
                elif agg_mode == "mean":
                    batch_dense_embeds = (
                        last_hidden_state * batch_tokenized_names["attention_mask"].unsqueeze(-1)
                    ).sum(1) / batch_tokenized_names["attention_mask"].sum(-1).unsqueeze(-1)
                else:
                    print("no such agg_mode:", agg_mode)

                batch_dense_embeds = batch_dense_embeds.cpu().detach().numpy()
                dense_embeds[start:end, :] = batch_dense_embeds
        if memory_map_file:
            dense_embeds.flush()
            dense_embeds = np.memmap(memory_map_file, dtype="float32", mode="r", shape=shape)

        return dense_embeds
