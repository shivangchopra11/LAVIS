import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer

from lavis.common.registry import registry
from lavis.models.beit_models.beit3_wrapper import BEiT3Wrapper, _get_base_config, _get_large_config
from lavis.models.beit_models.utils import merge_batch_tensors_by_dict_key, load_state_dict

class TwoLayerMLP(nn.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features, 
            out_features, 
            norm_layer, 
            norm_input=True, 
    ):
        super().__init__()
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output

@registry.register_model("beit3_vqa")
class BEiT3ForVisualQuestionAnswering(BEiT3Wrapper):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "vqav2": "configs/models/beit3_vqav2.yaml",
    }
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForVisualQuestionAnswering, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim, 
            output_features=embed_dim, 
            norm_layer=norm_layer, 
        )
        self.pooler.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), 
            norm_layer(embed_dim * 2), 
            nn.GELU(), 
            nn.Linear(embed_dim * 2, num_classes), 
        )
        self.head.apply(self._init_weights)

        self.tokenizer = self.init_tokenizer()

        self.ans2label_file = '/home/shivang/Desktop/GaTech/thesis/LAVIS/data/coco/images/answer2label.txt'
        self.label2ans = []
        self.ans2label = {}
        with open(self.ans2label_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                ans = data["answer"]
                label = data["label"]
                label = int(label)
                assert label == i
                self.ans2label[ans] = i
                self.label2ans.append(ans)

    def forward(self, image, question, padding_mask, **kwargs):
        outputs = self.beit3(
            textual_tokens=question, 
            visual_tokens=image, 
            text_padding_position=padding_mask, 
        )
        x = outputs["encoder_out"]
        cls_rep = self.pooler(x)
        return self.head(cls_rep)
    
    @classmethod
    def init_tokenizer(cls):
        tokenizer = XLMRobertaTokenizer("LAVIS/lavis/models/beit_models/beit3.spm")
        return tokenizer
    
    @classmethod
    def from_config(cls, cfg=None):
        args = _get_base_config(img_size=480)

        args.normalize_output = False
        model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129)
        state_dict = torch.load(cfg.pretrained_path)
        load_state_dict(model, state_dict['model'])
        return model
    
    def predict_answers(
        self,
        samples,
        device,
        inference_method="rank",
        **kwargs
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (str or [str]): String or a list of strings, each string is a question.
                                             The number of questions must be equal to the batch size. If a single string, will be converted to a list of string, with length 1 first.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            inference_method (str): Inference method. One of "rank", "generate".
                - If "rank", the model will return answers with the highest probability from the answer list.
                - If "generate", the model will generate answers.
            max_len (int): Maximum length of generated answers.
            min_len (int): Minimum length of generated answers.
            num_ans_candidates (int): Number of answer candidates, used to filter out answers with low probability.
            answer_list (list): A list of strings, each string is an answer.

        Returns:
            List: A list of strings, each string is an answer.

        Examples:
        ```python
            >>> from PIL import Image
            >>> from lavis.models import load_model_and_preprocess
            >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_vqa", "vqav2")
            >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
            >>> question = "Which city is this photo taken?"
            >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
            >>> question = txt_processors["eval"](question)
            >>> samples = {"image": image, "text_input": [question]}
            >>> answers = model.predict_answers(samples)
            >>> answers
            ['singapore']
            >>> answer_list = ["Singapore", "London", "Palo Alto", "Tokyo"]
            >>> answers = model.predict_answers(samples, answer_list=answer_list)
            >>> answers
            ['Singapore']
        ```
        """
        samples = merge_batch_tensors_by_dict_key(samples)
        image = samples['image'].to(device)
        question = samples['question'].to(device)
        padding_mask = samples['padding_mask'].to(device)
        logits = self.forward(image, question, padding_mask)
        print(logits)
        _, preds = logits.max(-1)
        answers = []
        for pred in preds:
            ans = self.label2ans[pred.item()]
            answers.append(ans)
        return answers

        