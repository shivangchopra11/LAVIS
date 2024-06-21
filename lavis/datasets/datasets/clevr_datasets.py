"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import json
import torch
from PIL import Image

from lavis.datasets.datasets.base_dataset import BaseDataset


class CLEVRVQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return None
        image_list, question_list, answer_list, weight_list = [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]

            answer_list.extend(answers)
            num_answers.append(len(answers))

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,
            "weight": weight_list,
            "n_answers": torch.LongTensor(num_answers),
        }

class CLEVRVQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root
        print(ann_paths)
        self.questions = json.load(open(ann_paths[0]))['questions']
        try:
            self.annotation = json.load(open(ann_paths[1]))['questions']
        except:
            self.annotation = json.load(open(ann_paths[1]))['annotations']

        self.vis_processor = vis_processor
        self.text_processor = text_processor


    def collater(self, samples):
        (
            image_list,
            question_list,
            question_id_list,
            answer_list,
            mask_list,
            instance_id_list,
            image_filename_list,
            raw_question_list
        ) = ([], [], [], [], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            answer_list.append(sample['answers'])
            mask_list.append(sample['mask'])
            instance_id_list.append(sample["question_id"])
            image_filename_list.append(sample["image_filename"])
            raw_question_list.append(sample["question"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "question_id": question_id_list,
            "answers": answer_list,
            "mask": mask_list,
            "instance_id": instance_id_list,
            "image_filename": image_filename_list,
            "question": raw_question_list
        }

    def __getitem__(self, index):

        ann = self.annotation[index]
        ques = self.questions[index]
        image_path = os.path.join(self.vis_root, ann["image_filename"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        try:
            question, mask = self.text_processor(ques["question"])
        except:
            question = self.text_processor(ques["question"])
            mask = []

        try:
            answer = ann['answer']
        except:
            answer = ann['answers']
            
        if "question_index" in ques.keys():
            ques_id = ques["question_index"]
        else:
            ques_id = ques["question_id"]

        return {
            "image": image,
            "text_input": question,
            "question_id": ques_id,
            "answers": answer,
            "mask": mask,
            "instance_id": ques_id,
            "image_filename": ann["image_filename"],
            "question": ques['question']
        }

