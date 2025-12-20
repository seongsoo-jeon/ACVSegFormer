import torch
import torch.nn
import os
import numpy as np
from mmcv import Config
import argparse
from utils import pyutils
from utility import mask_iou, Eval_Fmeasure, save_mask
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset


def main():
    # logger
    logger = getLogger(None, __name__)
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    logger.info(cfg.pretty_text)

    # model
    model = build_model(**cfg.model)
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    logger.info('Load trained model %s' % args.weights)

    # Test data
    test_dataset = build_dataset(**cfg.dataset.test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.dataset.test.batch_size,
                                                  shuffle=False,
                                                  num_workers=cfg.process.num_works,
                                                  pin_memory=True)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # Test
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            images, audio_feat, gt_mask, video_names = batch_data

            images = images.cuda()
            audio_feat = audio_feat.cuda()
            gt_mask = gt_mask.cuda()
            batch, frames, channels, height, width = images.shape
            images = images.view(batch * frames, channels, height, width)
            gt_mask = gt_mask.view(batch * frames, height, width)
            audio_feat = audio_feat.view(-1, audio_feat.shape[2],
                                         audio_feat.shape[3], audio_feat.shape[4])

            pred_mask, _ = model(audio_feat, images)

            if args.save_pred_mask:
                mask_save_path = os.path.join(
                    args.save_dir, dir_name, 'pred_masks')
                save_mask(pred_mask.squeeze(1), mask_save_path, video_names)

            miou = mask_iou(pred_mask.squeeze(1), gt_mask)
            avg_meter_miou.add({'miou': miou})
            f_score = Eval_Fmeasure(pred_mask.squeeze(1), gt_mask)
            avg_meter_F.add({'F_score': f_score})
            logger.info('batch_idx: {}, iou: {}, F_score: {}'.format(
                batch_idx, miou, f_score))
        miou = (avg_meter_miou.pop('miou'))
        f_score = (avg_meter_F.pop('F_score'))
        logger.info(f'test miou: {miou.item()}')
        logger.info(f'test F_score: {f_score}')
        logger.info('test miou: {}, F_score: {}'.format(miou.item(), f_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('weights', type=str, help='model weights path')
    parser.add_argument("--save_pred_mask", action='store_true',
                        default=False, help="save predited masks or not")
    parser.add_argument('--save_dir', type=str,
                        default='work_dir', help='save path')

    args = parser.parse_args()
    main()