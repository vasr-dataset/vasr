# ------------------------------Imports------------------------------
import os
import os.path
import torch
from PIL import Image
from collections import defaultdict
from experiments.config import IMAGES_PATH

# ------------------------------Constants------------------------------

INPUT_NAMES = ['A', 'B', 'C']
timm_models = ['vit', 'swin', 'deit', 'convnext']

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------Models------------------------------
class ZeroShot:

    def __init__(self, model, model_type, core_model_preprocess_func):
        """
        Parameters
        ----------
        model : Pretrained model for feature extraction
        model_type : (str) The name of the model {'vit', 'swin', 'deit', 'efficientnet', 'regnety'}
        core_model_preprocess_func : A torchvision transform that converts a PIL image into a tensor that the returned
        model can take as its input
        """

        self.model = model
        self.model_type = model_type
        if core_model_preprocess_func:
            self.core_model_preprocess_func = core_model_preprocess_func

    def get_scores(self, all_image_features):
        """

        Parameters
        ----------
        all_image_features : (dict) key is the image file name and the value is a preprocessed image Tensor

        Returns
        -------
        (dict) where the key is the image file name and the value is the list cosine scores for each candidate

        """
        scores = defaultdict(list)
        candidates = all_image_features['candidates']

        with torch.no_grad():
            chosen_features_list = {img_name: self.forward_core_model(all_image_features[img_name])
                                    for img_name in INPUT_NAMES}

            for k, im in enumerate(candidates):
                D_features = self.forward_core_model(im)

                for img_name, features in chosen_features_list.items():
                    score = self.get_img_cosine_similarity(D_features, features)
                    scores[img_name].append(float(score))

        return {img_name: scores[img_name] for img_name in INPUT_NAMES}

    def get_analogies_scores(self, all_image_features, candidate_names):
        """
        Parameters
        ----------
        all_image_features : (dict)
        candidate_names : (list) name of the image files of the candidates

        Returns
        -------
        Cosine similarities between any given candidate to C+(B-A)

        """
        scores = {}
        candidates = all_image_features['candidates']

        with torch.no_grad():
            all_inp_feats = {img_name: self.forward_core_model(all_image_features[img_name])
                             for img_name in INPUT_NAMES}

            # The goal is to find candidate D ~ C + (B-A)
            C_plus_B_minus_A = all_inp_feats['C'] + all_inp_feats['B'] - all_inp_feats['A']

            # For each candidate D calculate cosine (D,C + (B-A))
            for k, (im, cand_name) in enumerate(zip(candidates, candidate_names)):
                D_features = self.forward_core_model(im)
                score = self.get_img_cosine_similarity(C_plus_B_minus_A, D_features)
                scores[cand_name] = float(score)

        return scores

    def forward_core_model(self, img):
        """
        Parameters
        ----------
        img : (Tensor) image to extract features from

        Returns
        -------
        Features extracted from the given image

        """

        if self.model_type in timm_models:
            x = self.model.forward_features(img)
            if self.model_type == 'convnext':
                x = self.model.head(x)
            return x
        else:
            raise Exception(f"Unknown model {self.model_type}")

    def preprocess(self, img):
        """
        Parameters
        ----------
        img : file name of the image

        Returns
        -------
        (Tensor) preprocessed image

        """
        img_path = os.path.join(IMAGES_PATH, img)
        if self.model_type in timm_models:
            img_rgb = Image.open(img_path).convert('RGB')
            img_preprocessed = self.core_model_preprocess_func(img_rgb).unsqueeze(0).to(device)
            return img_preprocessed
        else:
            raise Exception(f"Unknown model {self.model_type}")

    @staticmethod
    def get_img_cosine_similarity(im1_feats, im2_feats):
        """
        Parameters
        ----------
        im1_feats : (Tensor) shape [1,embedding size]
        im2_feats : (Tensor) shape [1,embedding size]

        Returns
        -------
        Cosine similarity between the two given vectors

        """

        # Normalize the features
        im1_feats /= im1_feats.norm(dim=-1, keepdim=True)
        im2_feats /= im2_feats.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity
        similarity = im2_feats.detach().cpu().numpy() @ im1_feats.detach().cpu().numpy().T
        return similarity
