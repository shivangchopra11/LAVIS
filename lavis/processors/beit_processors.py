"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from transformers import XLMRobertaTokenizer


@registry.register_processor("beit_question")
class BeitQuestionProcessor(BaseProcessor):
    def __init__(self):
        self.tokenizer = XLMRobertaTokenizer("LAVIS/lavis/models/beit_models/beit3.spm")
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.num_max_bpe_tokens = 64

    def __call__(self, question):
        return self.pre_question(question)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def pre_question(self, question):
        tokens = self.tokenizer.tokenize(question)
        max_len = self.num_max_bpe_tokens
        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens = [self.bos_token_id] + token_ids[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask

