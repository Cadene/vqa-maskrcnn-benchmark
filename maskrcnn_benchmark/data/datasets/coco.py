# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
import numpy as np
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, ann_file, root, remove_images_without_annotations,
                 transforms=None, opencv_loading=False):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

        # Ideally this should be in an inherited subclass, but I'm lazy rn
        # Visual Genome training with attribute head
        self.has_attributes = "attCategories" in self.coco.dataset

        if self.has_attributes:
            self.attribute_ids = [x["id"] for x in self.coco.dataset["attCategories"]]

            self.json_attribute_id_to_contiguous_id = {
                v: i + 1 for i, v in enumerate(self.attribute_ids)
            }
            self.contiguous_attribute_id_to_json_id = {
                v: k for k, v in self.json_attribute_id_to_contiguous_id.items()
            }

            self.max_attributes_per_ins = 16
            self.num_attributes = 400

        self.opencv_loading = opencv_loading


    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        # VG related genome stuff
        attributes = -torch.ones((len(boxes), self.max_attributes_per_ins), dtype=torch.long)
        for idx, obj in enumerate(anno):
            if "attribute_ids" in obj:
                for jdx, att in enumerate(obj["attribute_ids"]):
                    attributes[idx, jdx] = att
        target.add_field("attributes", attributes)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            if self.opencv_loading:
                ## TEST only, Mimic Detectron Behaviour
                im = np.array(img).astype(np.float32)
                im = im[:, :, ::-1] 
                im -= self.transforms.transforms[-1].mean
                im_shape = im.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                im_scale = float(800) / float(im_size_min)
                # Prevent the biggest axis from being more than max_size
                if np.round(im_scale * im_size_max) > 1333:
                    im_scale = float(1333) / float(im_size_max)
                im = cv2.resize(
                    im,
                    None,
                    None,
                    fx=im_scale,
                    fy=im_scale,
                    interpolation=cv2.INTER_LINEAR
                )
                img = torch.from_numpy(im).permute(2, 0, 1)
                target.add_field("im_scale", im_scale)
            else:
                img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
