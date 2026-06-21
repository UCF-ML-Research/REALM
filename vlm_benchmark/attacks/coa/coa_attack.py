"""Chain of Attack (CoA) implementation for VLM benchmark (CVPR 2025)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from .core.clipcap import ClipCaptionModel, generate_cap

from ..base_attack import BaseAttack, AttackConfig, AttackResult
from ...data.base_dataset import Sample
from .data.coa_dataset import CoADataset


@dataclass
class COAAttackConfig(AttackConfig):
    """Configuration for Chain of Attack."""

    epsilon: float = 16.0
    max_iterations: int = 100
    alpha: float = 1.0

    clip_model_name: str = "ViT-B/32"
    input_res: int = 224

    clipcap_weights_path: Optional[str] = None
    prefix_length: int = 10

    fusion_type: str = "add_weight"
    a_weight: float = 0.3
    p_neg: float = 0.7

    use_caption_speedup: bool = False
    caption_update_steps: int = 1

    target_images_dir: Optional[str] = None
    target_captions_path: Optional[str] = None
    clean_captions_path: Optional[str] = None
    clean_images_dir: Optional[str] = None
    target_strategy: str = "stop_sign"


class COAAttack(BaseAttack):
    """Chain of Attack wrapper for VLM benchmark."""

    def __init__(self, config: COAAttackConfig):
        super().__init__(config)
        self.config: COAAttackConfig = config

        # Auto-set ClipCap weights path if not provided
        base_dir = Path(__file__).parent
        if config.clipcap_weights_path is None:
            config.clipcap_weights_path = str(
                base_dir / "assets" / "conceptual_weights.pt"
            )

        # Verify weights exist
        if not Path(config.clipcap_weights_path).exists():
            raise FileNotFoundError(
                f"ClipCap weights not found at {config.clipcap_weights_path}"
            )

        self._models_initialized = False
        self._caption_model_initialized = False
        self.to_pil = T.ToPILImage()

        self.clean_captions_cache: Dict[str, str] = {}
        self.target_captions_cache: Dict[str, str] = {}
        self.captions_modified = False
        self._target_caption_warned = False

    def _initialize_models(self):
        """Lazy load CLIP + ClipCap models by importing from legacy code."""
        if self._models_initialized:
            return

        print("Initializing CoA models (CLIP + ClipCap)...")

        import clip
        from transformers import GPT2Tokenizer

        print(f"  Loading CLIP model: {self.config.clip_model_name}")
        self.clip_model, self.clipcap_preprocess = clip.load(
            self.config.clip_model_name, device=self.config.device, jit=False
        )
        self.clip_model.eval()

        # Custom CLIP preprocessing matching legacy Chain_of_Attack/train.py
        self.clip_preprocess = T.Compose([
            T.Resize(
                self.clip_model.visual.input_resolution,
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True
            ),
            T.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            T.CenterCrop(self.clip_model.visual.input_resolution),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            ),
        ])

        print(f"  Loading ClipCap from: {self.config.clipcap_weights_path}")
        self.cap_model = ClipCaptionModel(prefix_length=self.config.prefix_length)
        self.cap_model.load_state_dict(
            torch.load(self.config.clipcap_weights_path, map_location="cpu"),
            strict=False,
        )
        self.cap_model = self.cap_model.eval().to(self.config.device)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.generate_cap = generate_cap

        self.clip_module = clip

        self._models_initialized = True
        print("✓ CoA models initialized")

    def _initialize_caption_model(self):
        """Lazy load Qwen3-VL for automatic caption generation."""
        if self._caption_model_initialized:
            return

        print("Initializing Qwen2.5-VL-3B for caption generation...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        self.caption_gen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            dtype=torch.bfloat16,
        ).to(self.config.device)
        self.caption_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self._process_vision_info = process_vision_info

        self._caption_model_initialized = True
        print("✓ Caption generation model initialized")

    def _generate_caption(self, image_path: str) -> str:
        """Generate a caption for an image using Qwen-VL."""
        if image_path in self.clean_captions_cache:
            return self.clean_captions_cache[image_path]

        self._initialize_caption_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence in English, focusing on the main objects and scene.",
                    },
                ],
            }
        ]

        text = self.caption_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.caption_processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.caption_gen_model.device)

        with torch.no_grad():
            generated_ids = self.caption_gen_model.generate(**inputs, max_new_tokens=128, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = self.caption_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

        # Fall back to OpenAI if caption contains non-ASCII (e.g. Chinese from Qwen)
        import re
        if re.search(r'[^\x00-\x7F]', caption):
            caption = self._gpt_caption(image_path, "clean")

        self.clean_captions_cache[image_path] = caption
        self.captions_modified = True
        print(f"  Generated caption: {caption[:60]}...")

        return caption

    def _generate_target_caption(self, image_path: str) -> str:
        """Generate a caption for a target image to guide the attack."""
        if image_path in self.target_captions_cache:
            return self.target_captions_cache[image_path]

        self._initialize_caption_model()

        if not self._target_caption_warned:
            print(
                "\n  WARNING: Auto-generating target captions from target images.\n"
                "  For stronger attacks, provide manual captions via --target_captions\n"
                "  that express adversarial intent (what the VLM should perceive)\n"
                "  rather than plain visual descriptions.\n"
            )
            self._target_caption_warned = True

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {
                        "type": "text",
                        "text": (
                            "Look at this image and write ONE short sentence describing "
                            "its main subject as if it were prominently present in a real "
                            "scene. Focus on what an observer would notice and how it is "
                            "significant — not just its visual appearance.\n\n"
                            "Good examples:\n"
                            '- Stop sign → "A stop sign is clearly visible ahead."\n'
                            '- Cat on a table → "A cat is sitting right in the middle of the table."\n'
                            '- Fire extinguisher → "A fire extinguisher is mounted on the wall nearby."\n'
                            '- Warning label → "A warning label indicates this area is hazardous."\n\n'
                            "Write only the caption, no explanation."
                        ),
                    },
                ],
            }
        ]

        text = self.caption_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.caption_processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.caption_gen_model.device)

        with torch.no_grad():
            generated_ids = self.caption_gen_model.generate(**inputs, max_new_tokens=64, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = self.caption_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip().strip('"').strip("'")

        # Fall back to OpenAI if caption contains non-ASCII (e.g. Chinese from Qwen)
        import re
        if re.search(r'[^\x00-\x7F]', caption):
            caption = self._gpt_caption(image_path, "target")

        self.target_captions_cache[image_path] = caption
        self.captions_modified = True
        print(f"  Generated target caption: {caption}")

        return caption

    def _gpt_caption(self, image_path: str, mode: str = "clean") -> str:
        """Fallback caption generation using GPT-5-mini via OpenAI API."""
        import base64
        from openai import OpenAI

        client = OpenAI()  # uses OPENAI_API_KEY
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        suffix = Path(image_path).suffix.lstrip(".")
        mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"

        if mode == "target":
            text = (
                "Look at this image and write ONE short sentence describing "
                "its main subject as if it were prominently present in a real scene. "
                "Write only the caption in English, no explanation."
            )
        else:
            text = "Describe this image in one sentence in English, focusing on the main objects and scene."

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": text},
                ],
            }],
            max_tokens=128,
        )
        return resp.choices[0].message.content.strip().strip('"').strip("'")

    def save_captions(
        self,
        images_dir: str,
        output_path: str,
        captions_cache: Optional[Dict[str, str]] = None,
    ):
        """Save generated captions to file for reuse."""
        cache = captions_cache if captions_cache is not None else self.clean_captions_cache
        if not cache:
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        images_dir = Path(images_dir)
        image_paths = sorted(
            list(images_dir.glob("*.jpg"))
            + list(images_dir.glob("*.png"))
            + list(images_dir.glob("*.jpeg"))
        )

        existing_captions = {}
        if output_path.exists():
            with open(output_path, "r") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if i < len(image_paths) and line.strip():
                    existing_captions[str(image_paths[i])] = line.strip()

        all_captions = {**existing_captions, **cache}

        captions = []
        for img_path in image_paths:
            img_path_str = str(img_path)
            if img_path_str in all_captions:
                captions.append(all_captions[img_path_str])
            else:
                break  # stop at first missing caption to keep file consistent

        with open(output_path, "w") as f:
            for caption in captions:
                f.write(caption + "\n")

        print(f"\n✓ Saved {len(captions)} captions to {output_path} ({len(cache)} new)")
        self.captions_modified = False

    def _run_coa_attack(
        self,
        clean_img_tensor: torch.Tensor,
        clean_caption: str,
        tgt_img_tensor: torch.Tensor,
        tgt_caption: str,
    ) -> torch.Tensor:
        """Run CoA PGD attack, returning the adversarial image tensor [C, H, W] in [0, 255]."""
        image_org = clean_img_tensor.unsqueeze(0).to(self.config.device)
        image_tgt = tgt_img_tensor.unsqueeze(0).to(self.config.device)

        cle_text = self.clip_module.tokenize([clean_caption]).to(self.config.device)
        tgt_text = self.clip_module.tokenize([tgt_caption]).to(self.config.device)

        # Compute static target and clean features
        with torch.no_grad():
            tgt_image_features = self.clip_model.encode_image(self.clip_preprocess(image_tgt))
            tgt_image_features = tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True)

            tgt_text_features = self.clip_model.encode_text(tgt_text)
            tgt_text_features = tgt_text_features / tgt_text_features.norm(dim=1, keepdim=True)

            cle_image_features = self.clip_model.encode_image(self.clip_preprocess(image_org))
            cle_image_features = cle_image_features / cle_image_features.norm(dim=1, keepdim=True)

            cle_text_features = self.clip_model.encode_text(cle_text)
            cle_text_features = cle_text_features / cle_text_features.norm(dim=1, keepdim=True)

            a = self.config.a_weight
            fusion_type = self.config.fusion_type

            if fusion_type == "cat":
                cle_fused_embedding = torch.cat((cle_image_features, cle_text_features), dim=1)
                cle_fused_embedding = cle_fused_embedding / cle_fused_embedding.norm(dim=1, keepdim=True)
                tgt_fused_embedding = torch.cat((tgt_image_features, tgt_text_features), dim=1)
                tgt_fused_embedding = tgt_fused_embedding / tgt_fused_embedding.norm(dim=1, keepdim=True)
            elif fusion_type == "add_weight":
                cle_fused_embedding = a * cle_image_features + (1 - a) * cle_text_features
                cle_fused_embedding = cle_fused_embedding / cle_fused_embedding.norm(dim=1, keepdim=True)
                tgt_fused_embedding = a * tgt_image_features + (1 - a) * tgt_text_features
                tgt_fused_embedding = tgt_fused_embedding / tgt_fused_embedding.norm(dim=1, keepdim=True)
            elif fusion_type == "multiplication":
                tgt_fused_embedding = tgt_image_features * tgt_text_features
                tgt_fused_embedding = tgt_fused_embedding / tgt_fused_embedding.norm(dim=1, keepdim=True)
                cle_fused_embedding = cle_image_features * cle_text_features
                cle_fused_embedding = cle_fused_embedding / cle_fused_embedding.norm(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        # PGD attack loop
        delta = torch.zeros_like(image_org, requires_grad=True)
        epsilon = self.config.epsilon
        alpha = self.config.alpha
        last_caption_list = None

        for j in range(self.config.max_iterations):
            adv_image = image_org + delta

            adv_image_clone = adv_image.clone() / 255.0

            adv_image_clip = self.clip_preprocess(adv_image)

            # Regenerate the caption dynamically each step (or reuse if speedup enabled)
            if self.config.use_caption_speedup and j % self.config.caption_update_steps != 0:
                current_caption_list = last_caption_list
            else:
                current_caption_list = []
                image_pil = self.to_pil(adv_image_clone[0].cpu())
                processed_image = self.clipcap_preprocess(image_pil).unsqueeze(0).to(self.config.device)

                with torch.no_grad():
                    prefix = self.clip_model.encode_image(processed_image).to(
                        self.config.device, dtype=torch.float32
                    )
                    prefix_embed = self.cap_model.clip_project(prefix).reshape(
                        1, self.config.prefix_length, -1
                    )

                current_caption = self.generate_cap(self.cap_model, self.tokenizer, embed=prefix_embed)
                current_caption_list.append(current_caption)
                last_caption_list = current_caption_list

            adv_image_features = self.clip_model.encode_image(adv_image_clip)
            adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)

            cur_caption = self.clip_module.tokenize(current_caption_list).to(self.config.device)
            cur_text_features = self.clip_model.encode_text(cur_caption)
            cur_text_features = cur_text_features / cur_text_features.norm(dim=1, keepdim=True)

            if fusion_type == "cat":
                cur_fused_embedding = torch.cat((adv_image_features, cur_text_features), dim=1)
                cur_fused_embedding = cur_fused_embedding / cur_fused_embedding.norm(dim=1, keepdim=True)
            elif fusion_type == "add_weight":
                cur_fused_embedding = a * adv_image_features + (1 - a) * cur_text_features
                cur_fused_embedding = cur_fused_embedding / cur_fused_embedding.norm(dim=1, keepdim=True)
            elif fusion_type == "multiplication":
                cur_fused_embedding = adv_image_features * cur_text_features
                cur_fused_embedding = cur_fused_embedding / cur_fused_embedding.norm(dim=1, keepdim=True)

            # Triplet-like loss: pull toward clean, push toward target
            embedding_sim1 = torch.mean(torch.sum(cur_fused_embedding * cle_fused_embedding, dim=1))
            embedding_sim2 = torch.mean(torch.sum(cur_fused_embedding * tgt_fused_embedding, dim=1))

            p_neg = self.config.p_neg
            margin = 1 - p_neg
            loss = torch.mean(torch.relu(embedding_sim2 - p_neg * embedding_sim1 + margin))
            loss.backward()

            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data = d
            delta.grad.zero_()

            if j % 20 == 0:
                print(f"  PGD step {j}/{self.config.max_iterations}: loss={loss.item():.4f}, caption={current_caption_list[0][:50]}")

        adv_image_final = image_org + delta
        adv_image_final = torch.clamp(adv_image_final, 0, 255)
        return adv_image_final.squeeze(0).cpu()

    def generate(self, model, sample: Sample, **kwargs) -> AttackResult:
        """Generate a CoA adversarial example for the given sample."""
        self._initialize_models()

        if not hasattr(self, "_coa_dataset"):
            self._coa_dataset = CoADataset(
                clean_images_dir=self.config.clean_images_dir,
                clean_captions_path=self.config.clean_captions_path,
                target_images_dir=self.config.target_images_dir,
                target_captions_path=self.config.target_captions_path,
            )

        coa_dataset = self._coa_dataset
        sample_idx = kwargs.get("sample_idx")

        if sample_idx is None:
            raise ValueError("CoA attack requires sample_idx in kwargs")

        coa_sample = coa_dataset[sample_idx]

        # Resize to CLIP input resolution before attack (legacy behavior)
        clean_img = coa_sample["clean_image"]
        target_img = coa_sample["target_image"]

        resize_transform = T.Compose([
            T.Resize(self.config.input_res, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.config.input_res),
        ])

        clean_img = resize_transform(clean_img)
        target_img = resize_transform(target_img)

        clean_tensor = torch.from_numpy(np.array(clean_img)).permute(2, 0, 1).float()
        target_tensor = torch.from_numpy(np.array(target_img)).permute(2, 0, 1).float()

        clean_caption = coa_sample.get("clean_caption")
        target_caption = coa_sample.get("target_caption")

        # Prefer source/target labels: short, concept-focused, fit CLIP's 77-token window
        if not clean_caption or clean_caption == "":
            clean_caption = (
                sample.metadata.get("source_label")
                or sample.metadata.get("attack_source_text")
                or ""
            )
            if not clean_caption:
                print(f"\nAuto-generating clean caption for {coa_sample['image_name']}...")
                clean_caption = self._generate_caption(coa_sample["image_path"])

        if not target_caption or target_caption == "":
            target_caption = (
                sample.metadata.get("target_label")
                or sample.metadata.get("attack_target_text")
                or ""
            )
            if not target_caption:
                print(f"\nAuto-generating target caption for {coa_sample['image_name']}...")
                target_caption = self._generate_target_caption(coa_sample["target_image_path"])

        # Truncate captions to fit CLIP's 77-token context length
        _tok = self.clip_module.tokenize
        for _try_len in (200, 120, 60, 30):
            try:
                _tok([clean_caption[:_try_len]])
                clean_caption = clean_caption[:_try_len]
                break
            except RuntimeError:
                continue
        for _try_len in (200, 120, 60, 30):
            try:
                _tok([target_caption[:_try_len]])
                target_caption = target_caption[:_try_len]
                break
            except RuntimeError:
                continue

        print(f"\nRunning CoA attack on {coa_sample['image_name']}")
        print(f"  Clean caption: {clean_caption}")
        print(f"  Target caption: {target_caption}")

        adv_tensor = self._run_coa_attack(
            clean_tensor,
            clean_caption,
            target_tensor,
            target_caption,
        )

        adv_image = Image.fromarray(adv_tensor.permute(1, 2, 0).byte().cpu().numpy())

        perturbation = (adv_tensor - clean_tensor).abs()
        perturbation_norm = perturbation.max().item()

        if self.captions_modified:
            if self.config.clean_images_dir and self.config.clean_captions_path:
                self.save_captions(self.config.clean_images_dir, self.config.clean_captions_path)
            if self.target_captions_cache and self.config.target_images_dir:
                auto_path = str(Path(self.config.target_images_dir) / "auto_captions.txt")
                self.save_captions(
                    self.config.target_images_dir, auto_path,
                    captions_cache=self.target_captions_cache,
                )

        original_output = ""
        adversarial_output = ""

        return AttackResult(
            success=True,  # transfer attack; success determined downstream
            adversarial_sample=adv_image,
            original_output=original_output,
            adversarial_output=adversarial_output,
            perturbation_norm=perturbation_norm,
            queries=0,
            metadata={
                "attack": "coa",
                "clean_caption": clean_caption,
                "target_caption": target_caption,
                "pgd_steps": self.config.max_iterations,
                "epsilon": self.config.epsilon,
                "clean_caption_auto": clean_caption in self.clean_captions_cache,
                "target_caption_auto": target_caption in self.target_captions_cache.values(),
            },
        )
