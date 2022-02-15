import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Resnet18(nn.Module):
    """
    Embedding extraction using Resnet-18 backbone

    Parameters
    ----------
    embedding_size:
        Size of embedding vector.

    pretrained:
        Whether to use pretrained weight on ImageNet.
    """
    def __init__(self, embedding_size: int, pretrained=False):
        super().__init__()

        model = models.resnet18(pretrained=pretrained)
        # Features extraction layers without the last fully-connected
        self.features = nn.Sequential(*list(model.children())[:-1])
        # Embeddding layer
        self.embedding = nn.Sequential(
            nn.Linear(in_features=512, out_features=embedding_size)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding features from image.

        Parameters
        ----------
        image:
            RGB image [3 x H x W].

        Returns
        -------
        torch.Tensor
            Embedding vector
        """
        embedding: torch.Tensor = self.features(image)
        embedding = embedding.flatten(start_dim=1)

        embedding: torch.Tensor = self.embedding(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class Resnet34(nn.Module):
    """
    Embedding extraction using Resnet-34 backbone

    Parameters
    ----------
    embedding_size:
        Size of embedding vector.

    pretrained:
        Whether to use pretrained weight on ImageNet.
    """
    def __init__(self, embedding_size: int, pretrained=False):
        super().__init__()

        model = models.resnet34(pretrained=pretrained)
        # Features extraction layers without the last fully-connected
        self.features = nn.Sequential(*list(model.children())[:-1])
        # Embeddding layer
        self.embedding = nn.Sequential(
            nn.Linear(in_features=512, out_features=embedding_size)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding features from image.

        Parameters
        ----------
        image:
            RGB image [3 x H x W].

        Returns
        -------
        torch.Tensor
            Embedding vector
        """
        embedding: torch.Tensor = self.features(image)
        embedding = embedding.flatten(start_dim=1)

        embedding: torch.Tensor = self.embedding(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class Resnet50(nn.Module):
    """
    Embedding extraction using Resnet-50 backbone

    Parameters
    ----------
    embedding_size:
        Size of embedding vector.

    pretrained:
        Whether to use pretrained weight on ImageNet.
    """
    def __init__(self, embedding_size: int, pretrained=False):
        super().__init__()

        model = models.resnet50(pretrained=pretrained)
        # Features extraction layers without the last fully-connected
        self.features = nn.Sequential(*list(model.children())[:-1])
        # Embeddding layer
        self.embedding = nn.Sequential(
            nn.Linear(in_features=2048, out_features=embedding_size)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding features from image.

        Parameters
        ----------
        image:
            RGB image [3 x H x W].

        Returns
        -------
        torch.Tensor
            Embedding vector
        """
        embedding: torch.Tensor = self.features(image)
        embedding = embedding.flatten(start_dim=1)

        embedding: torch.Tensor = self.embedding(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class EmbedderModel(nn.Module):
    pass


from typing import List, Dict
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODELS_CACHE_PATH = '/projects/academic/kjoseph/navid/models/cache'


def tokenize_sentences(sentences: List[str], tokenizer_path=MODELS_CACHE_PATH, max_len=160, batch_size=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=MODELS_CACHE_PATH)
    tokenizer_results = []
    pbar = tqdm(total=len(sentences) // batch_size)
    i = 0
    while i < len(sentences):
        batch = sentences[i:i + batch_size]
        tokenizer_res = tokenizer(
            batch, return_tensors='pt', max_length=max_len, truncation=True, padding='max_length')
        tokenizer_results.append(tokenizer_res)
        i += batch_size
        pbar.update(1)
    return tokenizer_results


def get_batch_embeddings(tokenizer_results):
    i = 0
    embeddings = []
    bert = BertModel.from_pretrained('bert-base-uncased', cache_dir=MODELS_CACHE_PATH).to(device)
    bert.eval()

    for tokenizer_dict in tqdm(tokenizer_results):
        for key in tokenizer_dict:
            tokenizer_dict[key] = tokenizer_dict[key].to(device)

        with torch.no_grad():
            outputs = bert(**tokenizer_dict)
        embeddings.append(outputs.pooler_output.detach().cpu().numpy())

    return embeddings