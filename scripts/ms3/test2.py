import torch
import torch.nn
import os
from PIL import Image
import numpy as np
import json 
# --------------------
from mmcv import Config
import argparse
from utils import pyutils
from utility import mask_iou, Eval_Fmeasure
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset


def save_mask_as_png(mask_tensor, save_path):
    threshold = 0.5 
    mask_tensor = mask_tensor.squeeze() 

    if mask_tensor.is_cuda:
        mask_np = mask_tensor.cpu().numpy()
    else:
        mask_np = mask_tensor.numpy()
        
    mask_np = (mask_np < threshold).astype(np.uint8) * 255 
    mask_image = Image.fromarray(mask_np, mode='L')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    mask_image.save(save_path)


def main():
    logger = getLogger(None, __name__)
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    logger.info(f'Load config from {args.cfg}')

    cfg = Config.fromfile(args.cfg)
    
    if cfg.dataset.test.batch_size != 1:
        cfg.dataset.test.batch_size = 1 
    
    logger.info(cfg.pretty_text)

    # model
    model = build_model(**cfg.model)
    model.load_state_dict(torch.load(args.weights), strict=False) 
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
    
    miou_analysis_threshold = 5
    
    failed_batches = [] 
    
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    mask_save_root = os.path.join(args.save_dir, dir_name, 'predictions') 

    # Test
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            images, audio_feat, gt_mask, video_names = batch_data

            images = images.cuda()
            audio_feat = audio_feat.cuda()
            gt_mask = gt_mask.cuda()
            batch, frames, channels, height, width = images.shape
            total_frames = batch * frames

            video_name = video_names[0]

            images = images.view(total_frames, channels, height, width)
            gt_mask = gt_mask.view(total_frames, height, width)
            audio_feat = audio_feat.view(-1, audio_feat.shape[2],
                                         audio_feat.shape[3], audio_feat.shape[4])

            pred_mask, *_ = model(audio_feat, images)

            miou = mask_iou(pred_mask.squeeze(1), gt_mask)
            current_miou_value = miou.item()
            f_score = Eval_Fmeasure(pred_mask.squeeze(1), gt_mask)

            if current_miou_value > miou_analysis_threshold:

                base_video_pred_dir = os.path.join(mask_save_root, video_name)

                if os.path.exists(base_video_pred_dir):

                    logger.info('FAILED BATCH {} (mIoU: {:.4f} < {:.4f}). Saving to CONTRAST...'.format(
                        batch_idx, current_miou_value, miou_analysis_threshold))

                    failed_batches.append({
                        'iter': batch_idx,
                        'miou': current_miou_value,
                        'F_score': f_score,
                        'video_name': video_name
                    })
                    video_pred_dir = os.path.join(mask_save_root, 'contrast', video_name, 'pred')

                    for i in range(total_frames):
                        frame_idx = i + 1

                        file_name = f'{video_name}_frame_{frame_idx:03d}.png'
                        pred_mask_path = os.path.join(video_pred_dir, file_name)

                        save_mask_as_png(pred_mask[i], pred_mask_path)

                else:
                    logger.debug(f"Skipping save for {video_name}: Base folder not found and mIoU < {miou_analysis_threshold}.")
                    pass


            avg_meter_miou.add({'miou': miou})
            avg_meter_F.add({'F_score': f_score})
            logger.info('batch_idx: {}, iou: {:.4f}, F_score: {:.4f}'.format(
                batch_idx, current_miou_value, f_score))

        miou = (avg_meter_miou.pop('miou'))
        f_score = (avg_meter_F.pop('F_score'))

        logger.info('--- Test Finished ---')
        logger.info('Total Failed Batches (mIoU < {:.4f}): {}'.format(miou_analysis_threshold, len(failed_batches)))

        if failed_batches:
            failed_list_path = os.path.join(args.save_dir, dir_name, 'failed_batches.json')
            os.makedirs(os.path.dirname(failed_list_path), exist_ok=True)
            with open(failed_list_path, 'w') as f:
                json.dump(failed_batches, f, indent=4)
            logger.info('Failed batches list saved to: {}'.format(failed_list_path))

        logger.info('test miou: {:.4f}, F_score: {:.4f}'.format(miou.item(), f_score))


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