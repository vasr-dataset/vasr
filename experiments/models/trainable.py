import torch
from torch import nn

device_ids = [0, 1, 2, 3]
NUM_CANDIDATE = 4


class BaselineModel(nn.Module):
    def __init__(self, backend_model, args):
        super(BaselineModel, self).__init__()
        self.backend_model = backend_model
        embed_dim = backend_model.get_embed_dim()

        if args.model_description == 'arithmetics':
            linear_layer_images = 2
        elif args.model_description == 'arithmetics_dist':
            linear_layer_images = 2
        else:
            linear_layer_images = 4
        pair_embed_dim = linear_layer_images * embed_dim
        self.model_description = args.model_description
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_embed_dim),
            nn.Linear(pair_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_images, candidates):
        input_images_embedding, candidates_features = self.extract_input_features(input_images, candidates)
        x = torch.cat([input_images_embedding, candidates_features], dim=1)
        x = self.classifier(x)
        return x

    def extract_input_features(self, input_images, candidates):
        initial_candidates_features = [self.backend_model.forward_core_model(candidate) for candidate in candidates]
        inp_embeddings = {k: self.backend_model.forward_core_model(inp_img) for k, inp_img in input_images.items()}

        # initial_candidates_features = self.backend_model.forward_core_model(candidates)
        # inp_embeddings = {k: self.backend_model.forward_core_model(inp_img) for k, inp_img in zip(['A', 'B', 'C'], input_images)}

        if self.model_description == 'concatenation':
            """ 
            the candidates contain 512 image vector for each option
            inp_embeddings includes 512 image vector for each ['A','B','C']
            we concat inp_embeddings feats to a single vector of 512*3 (1536) 
            """
            input_images_features = torch.cat([inp_embeddings[k] for k in ['A', 'B', 'C']], dim=1)
            final_candidates_features = initial_candidates_features

        elif self.model_description == 'arithmetics':
            input_images_features = inp_embeddings['C'] + (inp_embeddings['B'] - inp_embeddings['A'])
            final_candidates_features = initial_candidates_features

        elif self.model_description == 'arithmetics_dist':
            input_images_features = inp_embeddings['B'] - inp_embeddings['A']
            final_candidates_features = [option_feat - inp_embeddings['C'] for option_feat in initial_candidates_features]

        else:
            raise Exception(f"Not implemented get_input_images_embedding")
        return input_images_features, final_candidates_features
