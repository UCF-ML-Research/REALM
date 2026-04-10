"""GradCAM mask generator for AdvDiffVLM spatial attention."""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


class GradCAMGenerator:
    """Generate GradCAM masks using ResNet50."""

    def __init__(
        self,
        layer_name: str = "layer4",
        resolution: int = 64,
        device: str = "cuda"
    ):
        self.device = device
        self.resolution = resolution

        # Load ResNet50
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        self.model.to(device)
        self.model.eval()

        # Register forward hook
        self.features = {}
        self.gradients = {}
        layer = getattr(self.model, layer_name)
        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)

        # Preprocessing
        self.preprocess = weights.transforms()

    def _forward_hook(self, module, input, output):
        """Store intermediate features."""
        self.features['value'] = output

    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients."""
        self.gradients['value'] = grad_output[0]

    def generate(
        self,
        image: Image.Image,
        class_label: int
    ) -> torch.Tensor:
        """Generate GradCAM mask for image at 64×64 resolution."""
        # Preprocess
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        img_tensor.requires_grad_(True)

        # Forward pass
        self.model.zero_grad()
        logits = self.model(img_tensor)

        # Backward from target class
        target = logits[0, class_label]
        target.backward()

        # Compute GradCAM
        features = self.features['value']
        gradients = self.gradients['value']
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to 64×64
        cam = F.interpolate(
            cam,
            size=(self.resolution, self.resolution),
            mode='bilinear',
            align_corners=False
        )

        # Normalize [0, 1]
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)

        return cam.detach()
