import asyncio
import os
import os.path as osp
from argparse import ArgumentParser
import cv2
import mmcv
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default=None, help='Output directory of visualized images')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if args.img.endswith("png"):
        imgs = [args.img]
    elif args.img.endswith("txt"):
        imgs = [line.strip() for line in open(args.img).readlines()]
    else:
        raise NotImplementedError

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    prog_bar = mmcv.ProgressBar(len(imgs))
    for img in imgs:
        result = inference_detector(model, img)
        # show the results
        out_file = osp.join(args.out_dir, osp.basename(img)) if args.out_dir else None
        img_result = show_result_pyplot(model, img, result, score_thr=args.score_thr)
        cv2.imwrite(out_file, img_result)

        bboxes = np.vstack(result)
        num_obj = len(bboxes)
        bbox = np.zeros([num_obj, 5], dtype=np.float)
        bbox[:, 1] = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        bbox[:,0] = (bboxes[:,1] + bboxes[:,3]) * 0.5
        bbox[:,3] = abs(bboxes[:,2] - bboxes[:,0])
        bbox[:,2] = abs(bboxes[:,3] - bboxes[:,1])
        bbox[:,4] = bboxes[:,4]

        idx = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        idx = np.concatenate(idx)+1
        if args.score_thr > 0:
            assert bbox.shape[1] == 5
            scores = bbox[:, -1]
            inds = scores > args.score_thr
            bbox = bbox[inds, :]
            idx = idx[inds]
        CATEGORIES = ["__background", "building", "street_lamp", "traffic_light", "street_sign", "bench", "postbox",
                      "trunk"]
        labels = [CATEGORIES[i] for i in idx.tolist()]
        np.savez('/home3/zjl/mmdetection-master/data/result/annotation/2015-10-30-13-52-14/' + osp.basename(img).split('.')[0], bbox = bbox, idx = idx, label_nm = labels)
        prog_bar.update()


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)