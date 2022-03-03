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

        else:
            linear_layer_images = 4
        pair_embed_dim = linear_layer_images * embed_dim // 2
        # pair_embed_dim =  embed_dim//2

        self.model_description = args.model_description
        # self.multihead_attn = nn.MultiheadAttention(embed_dim//2, 8)

        if not args.cheap_model:
            self.pairs_layer = nn.Sequential(
                nn.LayerNorm(pair_embed_dim),
                nn.Linear(pair_embed_dim, pair_embed_dim),
                nn.ReLU(),
            )
            # self.classifier = nn.Sequential(
            #     nn.Linear((1+NUM_CANDIDATE) * pair_embed_dim, 384),
            #     nn.ReLU(),
            #     nn.Linear(384, 384//2),
            #     nn.ReLU(),
            #     nn.Linear(384//2, NUM_CANDIDATE)
            # )
            # self.classifier = nn.Sequential(
            #     nn.LayerNorm(NUM_CANDIDATE * pair_embed_dim),
            #     nn.Linear(NUM_CANDIDATE * pair_embed_dim, 384),
            #     nn.ReLU(),
            #     nn.Linear(384, NUM_CANDIDATE)
            # )
            self.classifier = nn.Sequential(
                nn.LayerNorm(pair_embed_dim),
                nn.Linear(pair_embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        else:
            print(f'BUILDING CHEAP BACKEND')
            self.pairs_layer = nn.Sequential(
                nn.LayerNorm(pair_embed_dim),
                nn.Linear(pair_embed_dim, int(pair_embed_dim / 8)),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(NUM_CANDIDATE * int(pair_embed_dim / 8), int(384 / 2)),
                nn.ReLU(),
                nn.Linear(int(384 / 2), NUM_CANDIDATE)
            )


    def forward(self, input_images, candidates):
        input_images_embedding, candidates_features = self.extract_input_features(input_images, candidates)
        x = torch.cat([input_images_embedding, candidates_features], dim=1)
        x = self.classifier(x)
        return x
    # def forward(self, input_images, candidates):
    #     input_images_embedding, candidates_features = self.extract_input_features(input_images, candidates)
    #
    #     pairs = []
    #     # x = torch.cat([input_images_embedding]+ candidates_features, dim=1)
    #
    #     for cand_feat in candidates_features:
    #         input_option_pair = torch.cat([input_images_embedding, cand_feat], dim=1)
    #         input_option_pair = self.pairs_layer(input_option_pair)
    #         pairs.append(input_option_pair)
    #     x = torch.cat(pairs, dim=1)
    #
    #     x = self.classifier(x)
    #     return x

    #
    # def forward(self, input_images, candidates):
    #     input_images_embedding, candidates_features = self.extract_input_features(input_images, candidates)
    #
    #     pairs = []
    #     for cand_feat in candidates_features:
    #         input_option_pair = torch.stack([input_images_embedding, cand_feat])
    #         input_option_pair, attn_output_weights = self.multihead_attn(query=input_option_pair, key=input_option_pair,
    #                                                                      value=input_option_pair)
    #         input_option_pair = torch.cat(torch.unbind(input_option_pair), dim=1)
    #         pairs.append(input_option_pair)
    #     x = torch.cat(pairs, dim=1)
    #
    #     x = self.classifier(x)
    #     return x
    def extract_input_features(self, input_images, candidates):
        # initial_candidates_features = [self.backend_model.forward_core_model(candidate) for candidate in candidates]
        initial_candidates_features = self.backend_model.forward_core_model(candidates)


        # inp_embeddings = {k: self.backend_model.forward_core_model(inp_img) for k, inp_img in input_images.items()}
        # TODO
        inp_embeddings = {k: self.backend_model.forward_core_model(inp_img) for k, inp_img in zip(['A', 'B', 'C'], input_images)}

        if self.model_description == 'concatenation':
            """ 
            the candidates contain 512 image vector for each option
            inp_embeddings includes 512 image vector for each ['A','B','C']
            we concat inp_embeddings feats to a single vector of 512*3 (1536) 
            """
            input_images_features = torch.cat([inp_embeddings[k] for k in ['A', 'B', 'C']], dim=1)
            final_candidates_features = initial_candidates_features

        elif self.model_description == 'arithmetics':
            final_candidates_features, input_images_features = self.get_analogies_feats(inp_embeddings, initial_candidates_features)

        else:
            raise Exception(f"Not implemented get_input_images_embedding")
        return input_images_features, final_candidates_features

    def get_analogies_feats(self, all_inp_feats, initial_candidates_features):
        C_plus_B_minus_A = all_inp_feats['C'] + all_inp_feats['B'] - all_inp_feats['A']
        input_images_features = C_plus_B_minus_A
        return initial_candidates_features, input_images_features
