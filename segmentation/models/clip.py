import open_clip  # type: ignore
import torch as T
import torchvision.transforms as TVT  # type: ignore


def get_clip_model(model_name: str, model_weights_name: str, device: str | T.device) -> tuple[T.nn.Module, TVT.Compose, open_clip.tokenizer.SimpleTokenizer]:
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_weights_name)
    model = model.to(device).eval()  # type: ignore
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer  # type: ignore
