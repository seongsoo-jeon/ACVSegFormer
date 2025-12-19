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
import matplotlib.pyplot as plt



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
                                                  #batch_size=cfg.dataset.test.batch_size,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=cfg.process.num_works,
                                                  pin_memory=True)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # Test
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio, mask, video_name_list = batch_data

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            mask = mask.view(B * frame, H, W)
            audio = audio.view(-1, audio.shape[2],
                               audio.shape[3], audio.shape[4])

            output, mask_feature, attn_maps = model(audio, imgs)

            if len(attn_maps) > 0:
                visual_dir = os.path.join(args.save_dir, dir_name, 'attention_results')
                os.makedirs(visual_dir, exist_ok=True)
                
                raw_vid_name = video_name_list[0]
                clean_vid_name = raw_vid_name.replace('/', '_')

                last_attn = attn_maps[-1]
                q_idx = 0 
                T = 5

                cross_map = last_attn['cross'][0].cpu().numpy() # [Q, Total_L]
                
                try:
                    cross_map_part = cross_map[q_idx, :5120].reshape(T, 32, 32)
                    
                    fig, axes = plt.subplots(1, T, figsize=(20, 4))
                    for t in range(T):
                        attn_frame = cross_map_part[t]
                        attn_frame = (attn_frame - attn_frame.min()) / (attn_frame.max() - attn_frame.min() + 1e-8)
                        
                        axes[t].imshow(attn_frame, cmap='jet')
                        axes[t].set_title(f"Time {t}")
                        axes[t].axis('off')
                    
                    cross_save_path = os.path.join(visual_dir, f"{clean_vid_name}_cross_attn.png")
                    plt.suptitle(f"Cross-Attention: {raw_vid_name}", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(cross_save_path)
                    plt.close()
                    logger.info(f"Saved Cross-Attn: {cross_save_path}")

                except Exception as e:
                    logger.error(f"Cross-Attn 시각화 실패: {e}")

                self_map = last_attn['self'][0].cpu().numpy() # [Q, Q]
                
                plt.figure(figsize=(10, 8))
                plt.imshow(self_map, cmap='viridis')
                plt.colorbar()
                plt.title(f"Self-Attention Matrix\n{raw_vid_name}", fontsize=12)
                plt.xlabel("Key Queries")
                plt.ylabel("Query Queries")
                
                self_save_path = os.path.join(visual_dir, f"{clean_vid_name}_self_attn.png")
                plt.tight_layout()
                plt.savefig(self_save_path)
                plt.close()
                logger.info(f"Saved Self-Attn: {self_save_path}") 

            if args.save_pred_mask:
                mask_save_path = os.path.join(
                    args.save_dir, dir_name, 'pred_masks')
                save_mask(output.squeeze(1), mask_save_path, video_name_list)

            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(output.squeeze(1), mask)
            avg_meter_F.add({'F_score': F_score})
            logger.info('n_iter: {}, iou: {}, F_score: {}'.format(
                n_iter, miou, F_score))
        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        logger.info(f'test miou: {miou.item()}')
        logger.info(f'test F_score: {F_score}')
        logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))


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