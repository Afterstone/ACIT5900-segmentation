from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch as T
import torchvision.transforms as TVT
from open_clip import pretrained
from PIL import Image
from PIL.Image import Image as PIL_Image
from scipy.ndimage import gaussian_filter
from segment_anything import SamPredictor, sam_model_registry
from tqdm import trange

import segmentation.config as config
from segmentation.datasets import FoodSegDataset
from segmentation.xai import gradCAM


def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
        (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map


def viz_attn(img, attn_map, blur=True):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    # Rescale the attention map to match the image size.
    attn_map = np.array(Image.fromarray(attn_map).resize((img.shape[1], img.shape[0])))
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    return fig, axes


def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.


def print_pretrained_models():
    for mn, wn in pretrained.list_pretrained():
        print(f"Model:   {mn}")
        print(f"Weights: {wn}")
        print()


def get_clip_model(model_name: str, model_weights_name: str, device: str | T.device) -> tuple[T.nn.Module, TVT.Compose, open_clip.tokenizer.SimpleTokenizer]:
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_weights_name)
    model = model.to(device).eval()  # type: ignore
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer  # type: ignore


def get_sam_model(
    sam_checkpoint: str = "models/sam_vit_h_4b8939.pth",
    sam_model_type: str = "vit_h",
    device: str | T.device = "cuda",
) -> SamPredictor:
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.eval().to(device=device)  # type: ignore
    return SamPredictor(sam)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)


@dataclass
class ClipClassifierResult:
    texts: list[str]
    probs: T.Tensor
    indices: T.Tensor


class ClipClassifier:
    def __init__(
        self,
        model_name: str,
        model_weights_name: str,
        device: str | T.device = 'cuda',
    ):
        self.model_name = model_name
        self.model_weights_name = model_weights_name
        self.device = device

        self.model, self.preprocessor, self.tokenizer = get_clip_model(
            model_name,
            model_weights_name,
            device,
        )

        self.classes: list[str] = []
        self.tokenized_text: T.Tensor = T.tensor([])
        self.text_features: T.Tensor = T.tensor([])
        self.text_features_normed: T.Tensor = T.tensor([])

    def set_classes(self, classes: list[str]) -> None:
        self.classes = classes
        self.tokenized_text = self.tokenizer(classes)
        self.text_features = self.model.encode_text(self.tokenized_text.cuda())  # type: ignore
        self.text_features_normed = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def classify(self, image: PIL_Image, top_k: int) -> ClipClassifierResult:
        tensor = self.preprocessor(image).unsqueeze(0).cuda()  # type: ignore
        image_features = self.model.encode_image(tensor)
        image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)

        text_probs = self.text_features_normed @ image_features_normed.T
        topk_indices = text_probs.squeeze().argsort(descending=True)[:top_k]
        topk_text_probs = text_probs.squeeze()[topk_indices]

        return ClipClassifierResult(
            texts=[self.classes[i] for i in topk_indices],
            probs=topk_text_probs.cpu().detach(),
            indices=topk_indices.cpu().detach(),
        )


def main(
    clip_model_name: str,
    clip_model_weights_name: str,
    clip_model_cls_name: str,
    clip_model_cls_weights_name: str,
    sam_checkpoint: str,
    sam_model_type: str,
    foodseg103_root: Path,
    prefix: str = "The dish contains the following: ",
    device: str | T.device = 'cuda',
) -> None:
    print("Loading dataset...")
    ds_test = FoodSegDataset.load_pickle(foodseg103_root / 'processed_test')

    print(f"Loading CLIP model {clip_model_name}...")
    clip_model, clip_preprocess, clip_tokenizer = get_clip_model(clip_model_name, clip_model_weights_name, device)

    print(f"Loading SAM model {sam_model_type}...")
    sam_predictor = get_sam_model(sam_checkpoint=sam_checkpoint, sam_model_type=sam_model_type, device=device)

    texts = ds_test.category_df['category'].tolist()
    with T.no_grad(), T.cuda.amp.autocast():  # type: ignore
        tokenized_text = clip_tokenizer([f"{prefix}{t}" for t in texts])
        text_features = clip_model.encode_text(tokenized_text.cuda())  # type: ignore

        clip_classifier = ClipClassifier(clip_model_cls_name, clip_model_cls_weights_name, device)
        clip_classifier.set_classes(texts)

    for idx in range(2, len(ds_test)):
        with T.no_grad(), T.cuda.amp.autocast():  # type: ignore
            # Load the annotations.
            annotation_path = ds_test.annotations_paths[idx]
            with Image.open(annotation_path) as img:
                ann_tensor = T.tensor(np.array(img))

            # Load the image.
            image_path = ds_test.image_paths[idx]
            with Image.open(image_path) as img:
                cls_res = clip_classifier.classify(img, 10)
                top_texts = [texts[i] for i in cls_res.indices]
                top_text_features = text_features[cls_res.indices]
                topk_text_probs = cls_res.probs

                image: T.Tensor = clip_preprocess(img).unsqueeze(0).cuda()  # type: ignore

        with T.cuda.amp.autocast():  # type: ignore
            attn_maps: dict[str, np.ndarray] = defaultdict()
            model_visual = clip_model.visual  # type: ignore
            model_layers = model_visual.trunk.stages  # type: ignore
            for text, text_feature in zip(top_texts, top_text_features):
                text_feature = text_feature.clone().unsqueeze(0)
                attn_map = gradCAM(model_visual, image.clone(), text_feature, layer=model_layers[3])
                attn_maps[text] = attn_map.squeeze().detach().cpu().numpy()

        image_np = load_image(image_path, resize=256)

        n_cols, n_rows = 3, len(attn_maps.keys())
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
        fig.suptitle(f"Model: {clip_model_name}\nWeights: {clip_model_weights_name}")
        for i, (text_prob, (text, atmap)) in enumerate(zip(topk_text_probs, attn_maps.items())):
            atmap = ((atmap - atmap.mean()) / atmap.std()).clip(0, 1)
            atmap = np.array(Image.fromarray(atmap).resize((image_np.shape[1], image_np.shape[0])))
            ax_orig, ax_attn, ax_sam = axes[i]
            for ax in axes[i]:
                ax.axis("off")

            ax_orig.imshow(image_np)
            ax_orig.set_title("Original")

            # ax_attn.imshow(getAttMap(image_np, atmap))
            ax_attn.imshow(atmap, alpha=0.5, cmap="jet")
            ax_attn.set_title(f"{text} - {text_prob:.4f}")

            atmap_np = np.array(atmap)

            size = atmap_np.shape
            probs = atmap_np.ravel()
            probs -= probs.min()
            probs = probs ** 2
            probs /= probs.sum()  # type: ignore

            x = np.arange(len(probs))
            idx = np.random.choice(x, 3, p=probs, replace=False)  # type: ignore
            idx_X = (idx // size[1])
            idx_y = (idx % size[1])
            input_points = np.array([(int(y), int(x)) for x, y in zip(idx_X, idx_y)])  # type: ignore
            input_labels = np.array([1] * len(input_points))
            show_points(input_points, input_labels, ax_attn)

            image_sam = (image_np.copy() * 255.0).astype(np.uint8)
            sam_predictor.set_image(image_sam)
            masks, scores, logits = sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            ax_sam.imshow(image_np)
            best_mask = masks[np.argmax(scores), :, :]
            show_mask(best_mask[np.newaxis, :, :], ax_sam, random_color=True)
            show_points(input_points, input_labels, ax_sam)

            fig.savefig("gradcam_openclip_sam.png")
            print()

        return

    # TODO: WIP
    #  - Break out code
    #  - Improve performance
    #    - Scalene?
    #    - The process is currently very slow - debug
    #    - Might have issues with matplotlib
    #    - Need to benchmark model performance
    #    - Caching potential?
    #      - Image loading etc.
    #  - Run on test set


if __name__ == '__main__':
    main(
        clip_model_name=config.CLIP_MODEL_NAME,
        clip_model_weights_name=config.CLIP_MODEL_WEIGHTS_NAME,
        clip_model_cls_name=config.CLIP_CLASSIFIER_NAME,
        clip_model_cls_weights_name=config.CLIP_CLASSIFIER_WEIGHTS_NAME,
        sam_checkpoint=config.SAM_CHECKPOINT,
        sam_model_type=config.MODEL_TYPE,
        device=config.TORCH_DEVICE,
        foodseg103_root=config.FOODSEG103_ROOT,
    )
