import torch
import torch.nn
import os
from PIL import Image
import numpy as np
import json 
from mmcv import Config
import argparse
from utils import pyutils
from utility import mask_iou, Eval_Fmeasure, save_mask
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset


def save_mask_as_png(mask_tensor, save_path, threshold=0.5):
    mask_tensor = mask_tensor.squeeze() 
    if mask_tensor.is_cuda:
        mask_np = mask_tensor.cpu().numpy()
    else:
        mask_np = mask_tensor.numpy()
        
    mask_np = (mask_np > threshold).astype(np.uint8) * 255
    
    mask_image = Image.fromarray(mask_np, mode='L')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    mask_image.save(save_path)


def main():
    # logger
    logger = getLogger(None, __name__)
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    
    cfg.dataset.test.batch_size = 1 
    logger.info(cfg.pretty_text)

    model = build_model(**cfg.model)
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    logger.info('Load trained model %s' % args.weights)

    test_dataset = build_dataset(**cfg.dataset.test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.dataset.test.batch_size,
                                                  shuffle=False,
                                                  num_workers=cfg.process.num_works,
                                                  pin_memory=True)
    
    threshold = 0.5
    failed_batches = []
    
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    save_png_root = os.path.join(args.save_dir, dir_name, 'predictions')

    # Test
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            images, audio_feat, gt_mask, video_names = batch_data

            images = images.cuda()
            audio_feat = audio_feat.cuda()
            gt_mask = gt_mask.cuda()
            batch, frames, channels, height, width = images.shape
            total_frames = batch * frames

            images = images.view(total_frames, channels, height, width)
            gt_mask = gt_mask.view(total_frames, height, width)
            audio_feat = audio_feat.view(-1, audio_feat.shape[2],
                                         audio_feat.shape[3], audio_feat.shape[4])

            pred_mask, *_ = model(audio_feat, images)

            miou = mask_iou(pred_mask.squeeze(1), gt_mask)
            current_miou_value = miou.item()
            f_score = Eval_Fmeasure(pred_mask.squeeze(1), gt_mask)

            if args.save_pred_mask:
                mask_save_path = os.path.join(
                    args.save_dir, dir_name, 'pred_masks')
            if current_miou_value < threshold:
                logger.warning(f'FAILED BATCH {batch_idx} (mIoU: {current_miou_value:.4f}). Saving masks...')

                video_name = video_names[0]
                failed_batches.append({
                    'iter': batch_idx,
                    'miou': current_miou_value,
                    'F_score': f_score,
                    'video_name': video_name
                })

                for i in range(total_frames):
                    frame_idx = i + 1

                    file_name = f'frame_{frame_idx:03d}.png'

                    pred_mask_dir = os.path.join(save_png_root, video_name, 'pred')
                    pred_mask_path = os.path.join(pred_mask_dir, file_name)
                    save_mask_as_png(pred_mask[i], pred_mask_path)

                    gt_mask_dir = os.path.join(save_png_root, video_name, 'gt')
                    gt_mask_path = os.path.join(gt_mask_dir, file_name)
                    save_mask_as_png(gt_mask[i], gt_mask_path)

            avg_meter_miou.add({'miou': miou})
            avg_meter_F.add({'F_score': f_score})
            logger.info('batch_idx: {}, iou: {:.4f}, F_score: {:.4f}'.format(
                batch_idx, current_miou_value, f_score))

        miou = (avg_meter_miou.pop('miou'))
        f_score = (avg_meter_F.pop('F_score'))

        logger.info(f'--- Test Finished ---')
        logger.info(f'Total Batches (mIoU < {threshold}): {len(failed_batches)}')

        if failed_batches:
            failed_list_path = os.path.join(args.save_dir, dir_name, 'good_batches.json')
            os.makedirs(os.path.dirname(failed_list_path), exist_ok=True)
            with open(failed_list_path, 'w') as f:
                json.dump(failed_batches, f, indent=4)
            logger.info(f'batches list saved to: {failed_list_path}')

        logger.info(f'test miou: {miou.item():.4f}, F_score: {f_score:.4f}')


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
    