import timm
import torch
from torch import nn
from PIL import Image
import os
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from config import IMAGES_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"


class BackendModel:
    """
    BackendModel is the feature extraction model (e.g., VIT)
    BaselineModel is trained to solve VASR using the features extracted from the BackendModel
    """

    def __init__(self, args):
        self.model_description = args.model_description
        self.model_backend_type = args.model_backend_type
        self.backend_version = args.backend_version
        self.multi_gpu = args.multi_gpu
        model, core_model_preprocess_func = self.create_timm_model(self.backend_version)

        model = model.to(device)
        if self.multi_gpu:
            model = nn.DataParallel(model)
        print(f"Checking backend model cuda: {next(model.parameters()).is_cuda}")

        print(f"Freezing backend model params")
        for param in model.parameters():
            param.requires_grad = False

        self.core_model_preprocess_func = core_model_preprocess_func
        self.backend_model = model

    def get_embed_dim(self):

        if self.model_backend_type in {'convnext', 'swin'}:
            embed_dim = self.backend_model.num_features
        else:
            embed_dim = self.backend_model.module.embed_dim if self.multi_gpu else self.backend_model.embed_dim
        return embed_dim

    def load_and_process_img(self, img):
        img_path = os.path.join(IMAGES_PATH, img)
        img_rgb = Image.open(img_path).convert('RGB')
        img_preprocessed = self.core_model_preprocess_func(img_rgb)
        img_preprocessed = img_preprocessed.to(device)
        return img_preprocessed

    def forward_core_model(self, img):
        s = '/Users/eliyahustrugo/PycharmProjects/image_analogies_0/imsitu_data/tensor_data'
        x = []
        if not isinstance(img,list):
            img = list(img)
        for i in img:
            i = i.replace('jpg', 'pt')
            with open(os.path.join(s, i[0], i), 'rb') as f:
                x.append(torch.load(f))

        x = torch.stack(x).squeeze()
        return x
    # def forward_core_model(self, img):
    #     s = '/Users/eliyahustrugo/PycharmProjects/image_analogies_0/imsitu_data/tensor_data'
    #     x = []
    #     for i in img:
    #         i = i.replace('jpg', 'pt')
    #         with open(os.path.join(s, i[0], i), 'rb') as f:
    #             x.append(torch.load(f))
    #
    #     x = torch.stack(x).squeeze()
    #     return x

    # def forward_core_model(self, img):
    #
    #     x = self.backend_model.forward_features(img)
    #     if self.model_backend_type == 'convnext':
    #         x = self.backend_model.head(x)
    #     return x

    def create_timm_model(self, backend_version):

        model = timm.create_model(backend_version, pretrained=True)
        model.eval()
        vit_config = resolve_data_config({}, model=model)
        transform = create_transform(**vit_config)
        return model, transform
