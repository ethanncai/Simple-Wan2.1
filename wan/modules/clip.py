import torch
import open_clip
from torchvision.transforms.functional import resize, normalize
from ..utils.train_utils import load_and_preprocess_image

class ClipImageEncoder(torch.nn.Module):
    def __init__(self, model_name='ViT-B-32', pretrained='openai', device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.visual.output_tokens = True

        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False
        )

    def _resize_to_clip_input(self, x):
        """
        x: [B, C, H, W], dtype float, range [0, 1] or [0, 255]
        Returns: [B, C, 224, 224] normalized for CLIP
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Resize directly to 224x224 (matches open_clip behavior)
        x = resize(x, size=(224, 224), antialias=True)

        # 确保 mean/std 与 x 同设备、同 dtype
        mean = self.clip_mean.to(device=x.device, dtype=x.dtype)
        std = self.clip_std.to(device=x.device, dtype=x.dtype)

        x = (x - mean) / std
        return x

    def forward(self, x):
        # x: [B, C, T, H, W]
        assert x.ndim == 5 or x.ndim == 4

        if x.ndim == 4:
            x = x.unsqueeze(0)
        x = x[:, :, 0]  # [B, C, H, W]
        x = self._resize_to_clip_input(x)
        x = x.to(self.device)
        _, tokens = self.model.visual(x)
        cls_token = tokens[:, 0]  # [B, 768]
        return cls_token

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    encoder = ClipImageEncoder(model_name='ViT-B-32', pretrained='openai', device=device)
    encoder.eval()

    img_path = "/home/rapverse/workspace_junzhi/Wan2.1/wan/dataset/assets/test_init_frame.jpg"

    img_tensor = load_and_preprocess_image(img_path, target_size=(480, 832))  # (C, 1, H, W)
    print(f"Preprocessed image tensor shape: {img_tensor.shape}")

    input_tensor = img_tensor.unsqueeze(0)  # (1, C, 1, H, W)
    input_tensor = input_tensor.to(device)
    print(f"Input to encoder shape: {input_tensor.shape} on {input_tensor.device}")

    with torch.no_grad():
        output = encoder(input_tensor)

    print(output.shape)