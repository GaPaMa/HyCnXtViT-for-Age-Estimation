import timm
print(timm.list_models())
model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
print(model.head)