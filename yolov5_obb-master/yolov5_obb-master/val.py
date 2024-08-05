# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

from utils.rboxs_utils import poly2hbb, rbox2poly

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, scale_polys, xywh2xyxy, xyxy2xywh, non_max_suppression_obb)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh 对应宽高 用于归一化
    for *xyxy, conf, cls in predn.tolist():
        #将xyxy(左上角+右上角)格式转化为xywh
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
        #存入runs\detect\exp lables


# def save_one_json(predn, jdict, path, class_map):
def save_one_json(pred_hbbn, pred_polyn, jdict, path, class_map):
    """
    Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236, "poly": [...]}
    Args:
        pred_hbbn (tensor): (n, [poly, conf, cls]) 
        pred_polyn (tensor): (n, [xyxy, conf, cls])
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(pred_hbbn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(pred_polyn.tolist(), box.tolist()): #序列解包
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[-1]) + 1], # COCO's category_id start from 1, not 0
                      'bbox': [round(x, 1) for x in b],
                      'score': round(p[-2], 5),
                      'poly': [round(x, 1) for x in p[:8]],#新增
                      'file_name': path.stem})

#计算指标
def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    返回每个预测框在10个IOU阈值上是TP还是FP
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    #构建[pred_nums,10]全为flase的矩阵
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    #计算每个gt与pred的iou，shape为[gt_nums,pred_nums]
    iou = box_iou(labels[:, 1:], detections[:, :4])
    #iou超过阈值且类别正确，则为True，返回索引
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    #存在符合条件的预测框
    if x[0].shape[0]:#至少有一个TP
        #符合的位置构成矩阵，第一列为行索引，第二列为列索引，第三列为iou值a
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            '''
            argsort从小到大 [::-1]取反从大到小
            返回唯一值的索引，[0]返回唯一值[1]返回索引
            每个预测框最多出现一次，若一个预测框多个gt，取其最大iou
            '''
            # matches = matches[matches[:, 2].argsort()[::-1]]
            '''
            matches[:,0]获取iou矩阵gt唯一值，返回最大值索引，每个gt最多出现一次
            '''
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv #计算correct，获取匹配预测框的iou信息
    return correct


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.01,  # confidence threshold
        iou_thres=0.4,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        ):
    # Initialize/load model and set device 初始化加载模型
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories 创建目录
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:#不同
            model.model.half() if half else model.model.float()
        elif engine:
            batch_size = model.batch_size
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data 检查数据文件是否正常
        data = check_dataset(data)  # check

    # Configure  加载yaml配置
    model.eval()#模型转为测试模式
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # Dataloader  是不是训练
    if not training:
        model.warmup(imgsz=(1, 3, imgsz, imgsz), half=half)  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, names, single_cls, pad=pad, rect=pt,
                                       workers=workers, prefix=colorstr(f'{task}: '))[0]
    #初始化已完成测试图片
    seen = 0
    #存储混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=nc) 
    # names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    #tqdm进度条
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'HBBmAP@.5', '  HBBmAP@.5:.95')
    #初始化detection各个指标
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # loss = torch.zeros(3, device=device)
    loss = torch.zeros(4, device=device) #多了theta
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    '''开始验证'''
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) θ ∈ [-pi/2, pi/2)
        # shapes (tensor): (b, [(h_raw, w_raw), (hw_ratios, wh_paddings)])
        t1 = time_sync()
        if pt or jit or engine:
            #将图片拷贝到device
            im = im.to(device, non_blocking=True)
            #对target拷贝
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference 前向推理
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        #out 推理结果 可打印
        #train_out 训练结果

        # Loss 计算损失 cls obj box
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, theta

        # NMS 获取预测框
        # targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        # out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        # 进行非极大值抑制
        out = non_max_suppression_obb(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls) # list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        dt[2] += time_sync() - t3

        # Metrics 统计预测框信息
        for si, pred in enumerate(out): # pred (tensor): (n, [xylsθ, conf, cls])
            #获取第si张图片的gt标签信息
            labels = targets[targets[:, 0] == si, 1:7] # labels (tensor):(n_gt, [clsid cx cy l s theta]) θ[-pi/2, pi/2)
            nl = len(labels) #目标个数      
            tcls = labels[:, 0].tolist() if nl else []  # target class 目标类别
            path, shape = Path(paths[si]), shapes[si][0] # shape (tensor): (h_raw, w_raw)
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions #改动大
            if single_cls:
                # pred[:, 5] = 0
                pred[:, 6] = 0
            poly = rbox2poly(pred[:, :5]) # (n, 8)
            pred_poly = torch.cat((poly, pred[:, -2:]), dim=1) # (n, [poly, conf, cls])
            hbbox = xywh2xyxy(poly2hbb(pred_poly[:, :8])) # (n, [x1 y1 x2 y2])
            pred_hbb = torch.cat((hbbox, pred_poly[:, -2:]), dim=1) # (n, [xyxy, conf, cls]) 

            pred_polyn = pred_poly.clone() # predn (tensor): (n, [poly, conf, cls])
            scale_polys(im[si].shape[1:], pred_polyn[:, :8], shape, shapes[si][1])  # native-space pred
            hbboxn = xywh2xyxy(poly2hbb(pred_polyn[:, :8])) # (n, [x1 y1 x2 y2])
            pred_hbbn = torch.cat((hbboxn, pred_polyn[:, -2:]), dim=1) # (n, [xyxy, conf, cls]) native-space pred
            

            # Evaluate
            if nl:
                # tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                tpoly = rbox2poly(labels[:, 1:6]) # target poly
                tbox = xywh2xyxy(poly2hbb(tpoly)) # target  hbb boxes [xyxy]
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labels_hbbn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels (n, [cls xyxy])
                correct = process_batch(pred_hbbn, labels_hbbn, iouv)
                if plots:
                    confusion_matrix.process_batch(pred_hbbn, labels_hbbn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            # stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred_poly[:, 8].cpu(), pred_poly[:, 9].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt: # just save hbb pred results!
                save_one_txt(pred_hbbn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                # LOGGER.info('The horizontal prediction results has been saved in txt, which format is [cls cx cy w h /conf/]')
            if save_json: # save hbb pred results and poly pred results.
                save_one_json(pred_hbbn, pred_polyn, jdict, path, class_map)  # append to COCO-JSON dictionary
                # LOGGER.info('The hbb and obb results has been saved in json file')
            callbacks.run('on_val_image_end', pred_hbb, pred_hbbn, path, names, im[si])

        # Plot images 画出框 gt(ground truth)真实框 pred预测框
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()
        '''
        Thread()为创建一个新的线程来执行这个函数，函数为plots.py中的plot_image函数
        target：执行函数 args：传入参数 daemon：主线程结束后，创建的子线程Thread已经自动结束
        start()：启动线程  可断点
        '''
    # Compute metrics 计算指标
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results 打印日志
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_obb_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
            LOGGER.info('---------------------The hbb and obb results has been saved in json file-----------------------')

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/DroneVehicle_poly.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'E:\yolov5-master\yolov5_obb-master\yolov5_obb-master\\runs\\train\exp27\weights\\best.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        # if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
        if opt.conf_thres > 0.01:  
            LOGGER.info(f'WARNING: In oriented detection, confidence threshold {opt.conf_thres} >> 0.01 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
