from transformers import AutoModel, AutoConfig

model_name = "facebook/dinov2-small"
save_dir   = "assets/dinov2-small-local"

model = AutoModel.from_pretrained(model_name)
model.save_pretrained(save_dir)

print(f"[Utils]\t Download DINOv2-small to {save_dir}")