from transformers import AlbertModel, AlbertTokenizer, AutoModel, AutoTokenizer


def get_pretrained_albert() -> tuple[AlbertModel, AlbertTokenizer]:
    return AutoModel.from_pretrained('albert-base-v1'), \
           AutoTokenizer.from_pretrained('albert-base-v1')
