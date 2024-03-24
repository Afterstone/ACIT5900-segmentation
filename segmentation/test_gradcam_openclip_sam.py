from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open_clip  # type: ignore
import torch as T
import torchvision.transforms as TVT  # type: ignore
from open_clip import pretrained
from PIL import Image
from PIL.Image import Image as PIL_Image
from segment_anything import SamPredictor, sam_model_registry  # type: ignore
from tqdm import trange

import segmentation.config as config
from segmentation.datasets import FoodSegDataset
from segmentation.xai import gradCAM


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
    # TODO: Make an interface for the classifier.
    def __init__(
        self,
        model_name: str,
        model_weights_name: str,
        classes: list[str] = [],
        device: str | T.device = 'cuda',
    ):
        self.model_name = model_name
        self.model_weights_name = model_weights_name
        self.device = device

        self.model, self.preprocessor, self.tokenizer = get_clip_model(model_name, model_weights_name, device)

        self._set_classes(classes)

    def _set_classes(self, classes: list[str]) -> None:
        self.classes = classes
        self.tokenized_text = self.tokenizer(classes)
        self.text_features = self.model.encode_text(self.tokenized_text.to(self.device))  # type: ignore
        self.text_features_normed = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def classify(self, image: PIL_Image, top_k: int) -> ClipClassifierResult:
        tensor = self.preprocessor(image).unsqueeze(0).to(self.device)  # type: ignore
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


class ClipAttentionMapper:
    # TODO: Make an interface for the attention mapper.
    def __init__(
        self,
        model_name: str,
        model_weights_name: str,
        classes: list[str] = [],
        device: str | T.device = 'cuda',
    ):
        self.model_name = model_name
        self.model_weights_name = model_weights_name
        self.device = device

        self.model, self.preprocessor, self.tokenizer = get_clip_model(model_name, model_weights_name, device)

        self._set_classes(classes)

    def _set_classes(self, classes: list[str]) -> None:
        self.classes: list[str] = classes
        self.tokenized_text: T.Tensor = self.tokenizer(classes)
        self.text_features: T.Tensor = self.model.encode_text(self.tokenized_text.to(self.device))  # type: ignore
        self.text_features_normed: T.Tensor = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def get_attention_maps(
        self,
        image: PIL_Image,
        top_texts: list[str],
        top_text_features: T.Tensor,
        normalize_attention: bool = True,
    ) -> dict[str, np.ndarray]:
        img_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)  # type: ignore
        attn_maps: dict[str, np.ndarray] = defaultdict()
        model_visual = self.model.visual  # type: ignore
        # TODO: Variable layer selection? Inject selector?
        model_layer = model_visual.trunk.stages[3]  # type: ignore
        for text, text_feature in zip(top_texts, top_text_features):
            text_feature = text_feature.clone().unsqueeze(0)
            # TODO: XAI method should be injected.
            attn_map = gradCAM(model_visual, img_tensor, text_feature, layer=model_layer)
            if normalize_attention:
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            attn_maps[text] = attn_map.squeeze().detach().cpu().numpy()

        return attn_maps


def propose_points(attn_map: np.ndarray, n_points: int = 3, norm: int = 2) -> np.ndarray:
    if not attn_map.ndim == 2:
        raise ValueError("attn_map must be 2D")
    size = attn_map.shape
    probs = attn_map.ravel()
    probs -= probs.min()
    probs = probs ** norm
    probs /= probs.sum()  # type: ignore

    x = np.arange(len(probs))
    idx = np.random.choice(x, n_points, p=probs, replace=False)  # type: ignore
    idx_X = (idx // size[1])
    idx_y = (idx % size[1])
    input_points = np.array([(int(y), int(x)) for x, y in zip(idx_X, idx_y)])  # type: ignore

    return input_points


def extract_class_indices_from_annotations(annotations: T.Tensor) -> T.Tensor:
    return annotations.ravel().unique().squeeze()


def get_sparse_annotation_masks(annotations: T.Tensor) -> dict[int, T.Tensor]:
    masks: dict[int, T.Tensor] = {}
    indices = [int(i) for i in extract_class_indices_from_annotations(annotations)]
    for i in indices:
        masks[i] = (annotations == i).float()
    return masks


@dataclass
class ClipAttentionMapperConfig:
    model_name: str
    model_weights_name: str


@dataclass
class ClipClassifierConfig:
    model_name: str
    model_weights_name: str


@dataclass
class SamConfig:
    model_type: str
    checkpoint: str


def main(
    clip_attn_mapper_config: ClipAttentionMapperConfig,
    clip_cls_config: ClipClassifierConfig,
    sam_config: SamConfig,
    foodseg103_root: Path,
    prefix: str = "The dish contains the following: ",
    device: str | T.device = 'cuda',
) -> None:
    print("Loading dataset...")
    ds_test = FoodSegDataset.load_pickle(foodseg103_root / 'processed_test')
    texts = ds_test.category_df['category'].tolist()

    with T.no_grad(), T.cuda.amp.autocast():  # type: ignore
        print(f"Loading CLIP model {clip_attn_mapper_config.model_name}...")
        attn_mapper = ClipAttentionMapper(clip_attn_mapper_config.model_name,
                                          clip_attn_mapper_config.model_weights_name, texts, device)

        # print(f"Loading CLIP classifier model {clip_cls_config.model_name}...")
        # clip_classifier = ClipClassifier(clip_cls_config.model_name, clip_cls_config.model_weights_name, texts, device)

        print(f"Loading SAM model {sam_config.model_type}...")
        sam_predictor = get_sam_model(sam_checkpoint=sam_config.checkpoint,
                                      sam_model_type=sam_config.model_type, device=device)

    ious = defaultdict(list)
    for idx in trange(0, len(ds_test)):
        # Load the annotations.
        with Image.open(ds_test.annotations_paths[idx]) as img:
            ann_tensor = T.tensor(np.array(img))
            ann_indices = [int(i) for i in extract_class_indices_from_annotations(ann_tensor.unsqueeze(0))]
            ann_mask_lookup = get_sparse_annotation_masks(ann_tensor.unsqueeze(0))

        image_path = ds_test.image_paths[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((256, 256))
            image_np = np.array(img).astype(np.float32) / 255.

            with T.no_grad(), T.cuda.amp.autocast():  # type: ignore
                # cls_res = clip_classifier.classify(img, top_k=10)
                # indices = cls_res.indices

                indices_list = ann_indices
                top_text_features_list = [attn_mapper.text_features[int(i)] for i in indices_list]
                top_text_features = T.stack(top_text_features_list)
                top_texts = [texts[i] for i in indices_list]

            with T.cuda.amp.autocast():  # type: ignore
                attn_maps = attn_mapper.get_attention_maps(img, top_texts, top_text_features)

        image_sam = (image_np.copy() * 255.0).astype(np.uint8)
        sam_predictor.set_image(image_sam)
        for i, (_, atmap) in enumerate(attn_maps.items()):
            atmap = np.array(Image.fromarray(atmap).resize((image_np.shape[1], image_np.shape[0])))  # type: ignore
            input_points = propose_points(np.array(atmap), n_points=3)
            input_labels = np.array([1] * len(input_points))

            masks, scores, logits = sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            best_mask = masks[np.argmax(scores), :, :]
            best_mask_tensor = T.tensor(best_mask).ravel()

            annotation = ann_mask_lookup[indices_list[i]].squeeze()
            annotation = T.from_numpy(np.array(Image.fromarray(annotation.numpy()).resize(
                (image_np.shape[1], image_np.shape[0]))))
            annotation_bool = annotation > 0
            annotation_bool = annotation_bool.ravel()

            intersection = T.logical_and(best_mask_tensor, annotation_bool)
            union = T.logical_or(best_mask_tensor, annotation_bool)
            iou = T.true_divide(intersection.sum(), union.sum())

            ious[indices_list[i]].append(iou.item())

            if idx > 0 and idx % 100 == 0:
                means = []
                for i, iou_list in sorted(list(ious.items()), key=lambda x: x[0]):
                    means.append(np.mean(iou_list))
                    print(f"Class {i: 3d}: {means[-1]:.4f}")
                print("-"*10)
                print(f"mIOU: {np.mean(means):.4f}")
                print()

    means = []
    for i, iou_list in sorted(list(ious.items()), key=lambda x: x[0]):
        means.append(np.mean(iou_list))
        print(f"Class {i: 3d}: {means[-1]:.4f}")
    print("-"*10)
    print(f"mIOU: {np.mean(means):.4f}")
    print()

    # n_cols, n_rows = 3, len(attn_maps.keys())
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    # fig.suptitle(f"Model: {clip_model_name}\nWeights: {clip_model_weights_name}")
    # for i, (text_prob, (text, atmap)) in enumerate(zip(topk_text_probs, attn_maps.items())):
    #     ax_orig, ax_attn, ax_sam = axes[i]
    #     for ax in axes[i]:
    #         ax.axis("off")

    #     ax_orig.imshow(image_np)
    #     ax_orig.set_title("Original")

    #     atmap = np.array(Image.fromarray(atmap).resize((image_np.shape[1], image_np.shape[0])))
    #     ax_attn.imshow(atmap, alpha=0.5, cmap="jet")
    #     ax_attn.set_title(f"{text} - {text_prob:.4f}")

    #     input_points = propose_points(np.array(atmap), n_points=3)
    #     input_labels = np.array([1] * len(input_points))
    #     show_points(input_points, input_labels, ax_attn)

    #     image_sam = (image_np.copy() * 255.0).astype(np.uint8)
    #     sam_predictor.set_image(image_sam)
    #     masks, scores, logits = sam_predictor.predict(
    #         point_coords=input_points,
    #         point_labels=input_labels,
    #         multimask_output=True,
    #     )

    #     ax_sam.imshow(image_np)
    #     best_mask = masks[np.argmax(scores), :, :]
    #     show_mask(best_mask[np.newaxis, :, :], ax_sam, random_color=True)
    #     show_points(input_points, input_labels, ax_sam)

    #     fig.savefig("gradcam_openclip_sam.png")
    #     print()

    # return


if __name__ == '__main__':
    main(
        clip_attn_mapper_config=ClipAttentionMapperConfig(
            model_name=config.CLIP_MODEL_NAME,
            model_weights_name=config.CLIP_MODEL_WEIGHTS_NAME,
        ),
        clip_cls_config=ClipClassifierConfig(
            model_name=config.CLIP_CLASSIFIER_NAME,
            model_weights_name=config.CLIP_CLASSIFIER_WEIGHTS_NAME,
        ),
        sam_config=SamConfig(
            checkpoint=config.SAM_CHECKPOINT,
            model_type=config.MODEL_TYPE,
        ),
        device=config.TORCH_DEVICE,
        foodseg103_root=config.FOODSEG103_ROOT,
    )