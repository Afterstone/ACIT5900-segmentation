import typing as t
from collections import defaultdict
from dataclasses import dataclass
from math import isnan
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
import segmentation.xai as xai
from segmentation.datasets import FoodSegDataset
from segmentation.normalizers import (Identity, MinMaxNormalizer, Normalizer,
                                      PercentileThresholder,
                                      PolynomialStretcher,
                                      SequentialNormalizer, SoftmaxNormalizer,
                                      StandardNormalizer, Thresholder,
                                      ToProbabilities, UniformIfInvalid)


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
        cam_method: xai.BaseCam,
        classes: list[str] = [],
        device: str | T.device = 'cuda',
        normalizer: Normalizer = Identity(),
        epsilon: float = 1e-9,
    ):
        self.model_name = model_name
        self.model_weights_name = model_weights_name
        self.device = device
        self.cam_method = cam_method
        self.epsilon = epsilon

        self.normalizer = normalizer

        self.model, self.preprocessor, self.tokenizer = get_clip_model(model_name, model_weights_name, device)
        self._set_classes(classes)

        self.model_visual = self.model.visual  # type: ignore

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
    ) -> dict[str, np.ndarray]:
        img_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)  # type: ignore
        attn_maps: dict[str, np.ndarray] = defaultdict()

        for text, text_feature in zip(top_texts, top_text_features):
            text_feature = text_feature.clone().unsqueeze(0)
            attn_map = self.cam_method(self.model_visual, img_tensor, text_feature).detach()
            attn_map = self.normalizer.normalize(attn_map)
            attn_map_np = attn_map.squeeze().detach().cpu().numpy()

            if np.isnan(attn_map_np).any():
                raise ValueError(f"Encountered NaN in attention map for text: {text}")

            attn_maps[text] = attn_map_np

        return attn_maps


def propose_points(attn_map: np.ndarray, n_points: int) -> np.ndarray:
    if not attn_map.ndim == 2:
        raise ValueError("attn_map must be 2D")

    probs = attn_map.ravel()

    # Resizing might have caused some negative values.
    probs -= probs.min()
    probs /= probs.sum()

    if np.isnan(probs).any():
        probs = np.ones_like(probs) / len(probs)

    x = np.arange(len(probs))
    idx = np.random.choice(x, n_points, p=probs, replace=False)  # type: ignore
    size = attn_map.shape
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
    xai_method: str
    normalize_attention: str
    normalizer_norm: float

    def __post_init__(self):
        self.xai_method = self.xai_method.lower()
        self.normalize_attention = self.normalize_attention.lower()

    def get_xai_method(self) -> xai.BaseCam:
        def layers_extractor(model: T.nn.Module) -> list[T.nn.Module]:
            if self.model_name.lower().startswith("rn"):
                return [
                    model.layer1,
                    model.layer2,
                    model.layer3,
                    model.layer4,
                ]
            elif self.model_name.lower().startswith("convnext"):
                return model.trunk.stages
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

        if self.xai_method == "gradcam":
            return xai.GradCam(layers_extractor=layers_extractor)
        elif self.xai_method == "gradcampp":
            return xai.GradCamPP(layers_extractor=layers_extractor)
        elif self.xai_method == "layercam":
            return xai.LayerCam(layers_extractor=layers_extractor)
        elif self.xai_method == "uniform":
            return xai.UniformXai(layers_extractor=layers_extractor)
        else:
            raise ValueError(f"Unsupported XAI method: {self.xai_method}")

    def get_normalization_method(self) -> Normalizer:
        normalizer: Normalizer = Identity()
        thresholder: Normalizer = Identity()
        if self.normalize_attention == 'none':
            pass
        elif self.normalize_attention == "minmax":
            normalizer = MinMaxNormalizer()
            thresholder = Thresholder(0.5)
        elif self.normalize_attention == "standard":
            normalizer = StandardNormalizer()
            thresholder = Thresholder(0.0)
        elif self.normalize_attention == "softmax":
            normalizer = SoftmaxNormalizer()
        elif self.normalize_attention == "percentile":
            thresholder = PercentileThresholder(0.95)
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalize_attention}")

        return SequentialNormalizer([
            normalizer,
            thresholder,
            PolynomialStretcher(degree=self.normalizer_norm),
            ToProbabilities(),
            UniformIfInvalid(),
        ])


@dataclass
class ClipClassifierConfig:
    model_name: str
    model_weights_name: str


@dataclass
class SamConfig:
    model_type: str
    checkpoint: str
    proposer_n_points: int


@dataclass
class EvaluationResults:
    mIoU: float
    aAcc: float


def evaluate(
    clip_attn_mapper_config: ClipAttentionMapperConfig,
    clip_cls_config: ClipClassifierConfig,
    sam_config: SamConfig,
    dataset: FoodSegDataset,
    prefix: str = "The dish contains the following: ",
    device: str | T.device = 'cuda',
    print_results_interval: int = 10,
    progress_callback: t.Callable[[int, dict[str, str | float | int]], None] | None = None,
    progress_callback_interval: int = 10,
) -> EvaluationResults:
    print("Loading dataset...")
    texts = [f"{prefix}{x}" for x in dataset.category_df['category'].tolist()]

    with T.no_grad(), T.cuda.amp.autocast():  # type: ignore
        print(f"Loading CLIP model \"{clip_attn_mapper_config.model_name}\" with"
              f" weights \"{clip_attn_mapper_config.model_weights_name}\"...")

        attn_mapper = ClipAttentionMapper(
            model_name=clip_attn_mapper_config.model_name,
            model_weights_name=clip_attn_mapper_config.model_weights_name,
            cam_method=clip_attn_mapper_config.get_xai_method(),
            classes=texts,
            device=device,
            normalizer=clip_attn_mapper_config.get_normalization_method(),
        )

        # print(f"Loading CLIP classifier model \"{clip_cls_config.model_name}\" with "
        #       f"weights \"{clip_cls_config.model_weights_name}\"...")
        # clip_classifier = ClipClassifier(
        #     model_name=clip_cls_config.model_name,
        #     model_weights_name=clip_cls_config.model_weights_name,
        #     classes=texts,
        #     device=device
        # )

        print(f"Loading SAM model \"{sam_config.model_type}\" with checkpoint \"{sam_config.checkpoint}\"...")
        sam_predictor = get_sam_model(sam_checkpoint=sam_config.checkpoint,
                                      sam_model_type=sam_config.model_type, device=device)

    pixel_accs = defaultdict(list)
    ious = defaultdict(list)
    progbar = trange(0, len(dataset))
    for idx in progbar:
        # Load the annotations.
        with Image.open(dataset.annotations_paths[idx]) as img:
            ann_tensor = T.tensor(np.array(img))
            ann_indices = [int(i) for i in extract_class_indices_from_annotations(ann_tensor.unsqueeze(0))]
            ann_mask_lookup = get_sparse_annotation_masks(ann_tensor.unsqueeze(0))

        image_path = dataset.image_paths[idx]
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
                try:
                    attn_maps = attn_mapper.get_attention_maps(img, top_texts, top_text_features)
                except ValueError:
                    print(f"Encountered NaN when getting attention map for image {image_path}")
                    continue

        image_sam = (image_np.copy() * 255.0).astype(np.uint8)
        sam_predictor.set_image(image_sam)
        for i, (_, atmap) in enumerate(attn_maps.items()):
            atmap = np.array(Image.fromarray(atmap).resize((image_np.shape[1], image_np.shape[0])))  # type: ignore
            input_points = propose_points(
                np.array(atmap),
                n_points=sam_config.proposer_n_points,
            )
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

            index = indices_list[i]

            intersection = T.logical_and(best_mask_tensor, annotation_bool)
            union = T.logical_or(best_mask_tensor, annotation_bool)
            ious[index].append(T.true_divide(intersection.sum(), union.sum()).item())

            pixel_accs[index].append((best_mask_tensor == annotation_bool).float().mean().item())

        if idx > 0 and (
            (idx % print_results_interval == 0)
            or (progress_callback is not None and idx % progress_callback_interval == 0)
        ):
            iou_means = []
            for i, iou_list in sorted(list(ious.items()), key=lambda x: x[0]):
                iou_means.append(np.mean(iou_list))
            pixel_acc_means = []
            for i, acc_list in sorted(list(pixel_accs.items()), key=lambda x: x[0]):
                pixel_acc_means.append(np.mean(acc_list))

            miou = float(np.mean(iou_means))
            aacc = float(np.mean(pixel_acc_means))

            if idx > 0 and idx % print_results_interval == 0:
                progbar.set_postfix(mIOU=f"{miou:.4f}", aAcc=f"{aacc:.4f}")

            if progress_callback is not None:
                results: dict[str, str | float | int] = {
                    "miou": miou,
                    "aacc": aacc,
                }
                progress_callback(idx, results)

    iou_means = []
    for i, iou_list in sorted(list(ious.items()), key=lambda x: x[0]):
        iou_means.append(np.mean(iou_list))

    pixel_acc_means = []
    for i, acc_list in sorted(list(pixel_accs.items()), key=lambda x: x[0]):
        pixel_acc_means.append(np.mean(acc_list))

    print("-"*10)
    print(f"mIOU: {np.mean(iou_means):.4f}")
    print(f"aAcc: {np.mean(pixel_acc_means):.4f}")
    print()

    return EvaluationResults(
        mIoU=float(np.mean(iou_means)),
        aAcc=float(np.mean(pixel_acc_means)),
    )

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
    evaluate(
        clip_attn_mapper_config=ClipAttentionMapperConfig(
            model_name=config.CLIP_MODEL_NAME,
            model_weights_name=config.CLIP_MODEL_WEIGHTS_NAME,
            xai_method=config.XAI_METHOD,
            normalize_attention=config.CLIP_NORMALIZE_ATTENTION,
            normalizer_norm=config.CLIP_NORMALIZER_NORM,
        ),
        clip_cls_config=ClipClassifierConfig(
            model_name=config.CLIP_CLASSIFIER_NAME,
            model_weights_name=config.CLIP_CLASSIFIER_WEIGHTS_NAME,
        ),
        sam_config=SamConfig(
            checkpoint=config.SAM_CHECKPOINT,
            model_type=config.MODEL_TYPE,
            proposer_n_points=config.SAM_PROPOSER_N_POINTS,
        ),
        device=config.TORCH_DEVICE,
        dataset=FoodSegDataset.load_pickle(config.FOODSEG103_ROOT / 'processed_test'),
        print_results_interval=config.PRINT_RESULTS_EVERY_N_STEPS,
    )
