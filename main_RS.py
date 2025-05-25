import os
import sys
from config import parse_option
args, config = parse_option()
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.CUDA_VISIBLE_DEVICES_)
sys.path.append('.')

import time
import json
import random
import datetime
import numpy as np
import itertools
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import accuracy, AverageMeter
from models import build_model
from data import build_loader
from utils.lr_scheduler import build_scheduler, build_scheduler_fine_tuning
from utils.optimizer import build_optimizer, build_optimizer_fine_tuning
from utils.logger import create_logger
from utils.utils import NativeScalerWithGradNormCount
from utils.loss_func import ClipLoss, DIST, DistillKL
from utils.utils import Queue
from models.network import Text_Net
from models import clip
from sklearn.metrics import classification_report, recall_score, cohen_kappa_score, accuracy_score
import scipy.io as sio

def set_module_eval(module):
    for name, sub_module in module.named_children():
        if not any(param.requires_grad for param in sub_module.parameters(recurse=False)):
            sub_module.eval()
        else:
            sub_module.train()
        set_module_eval(sub_module)
    return module

def generate_all_masks(N):
    masks = []
    # for i in range(1, N):
    for i in range(N):
        for comb in itertools.combinations(range(N), i):
            mask = np.ones(N)
            mask[list(comb)] = 0
            masks.append(mask)
    return masks

def get_approx_uniform_masks(B, N):
    masks = generate_all_masks(N)
    num_masks = len(masks)
    full_repeats = B // num_masks
    remainder = B % num_masks
    chosen_masks = np.tile(masks, (full_repeats, 1))
    if remainder > 0:
        chosen_masks = np.vstack([chosen_masks, masks[:remainder]])
    np.random.shuffle(chosen_masks)
    return torch.tensor(chosen_masks, dtype=torch.float32)

def generate_mask_patch(P, mask_ratio=0.8):
    num_pixels = P * P
    num_masked_pixels = int(num_pixels * mask_ratio)
    mask_flat = torch.ones(num_pixels)
    mask_flat[:num_masked_pixels] = 0
    mask_flat = mask_flat[torch.randperm(num_pixels)]
    mask = mask_flat.view(P, P)
    return mask

@torch.no_grad()
def get_text_feature():
    if "Trento" in config.DATA.DATASET_NAME:
        text = [
            'A sample of Apple trees. Apple trees are deciduous fruit trees with medium-sized stature and broad, spreading canopies. They bear glossy green oval leaves with serrated edges. They thrive in orchards or farmlands, with the ground supporting their roots. Surrounding roads aid in transporting apples',
            'A sample of Buildings. Buildings vary in size, designs, and materials like concrete, brick, wood, and glass. They stand firmly on the ground with rooted foundations. Roads connect buildings and amenities for accessibility. In urban areas, buildings form cityscapes with roads and green spaces, including parks and woods.',
            'A sample of Ground. The ground, composed of soil, rocks, and geological materials, serves as the foundation for buildings, roads, and vineyards. It nurtures apple trees and plants in woods, supporting their growth. Roads connect vineyards and apple orchards, passing through various locations.',
            'A sample of Woods. The ground supports tree and vegetation growth in woods. Roads may pass through or surround woods. Woods are a habitat for wildlife, including pollinators that benefit apple trees and vineyards.',
            'A sample of Vineyard. Vineyards are cultivated areas dedicated to growing grapevines for wine production or other purposes. Ground supports grapevines in vineyards. Roads aid grape transportation. Vineyards border woods with diverse flora and fauna.',
            'A sample of Roads. Roads are designed pathways for vehicular and pedestrian traffic. They vary in size and construction materials, including asphalt, concrete, or grave. Roads connect landscapes - orchards, vineyards, and woods. ',
            ]
    elif "Houston" in config.DATA.DATASET_NAME:
        text = [
            'A sample of Health grass. Health grass is recognized by its vibrant green color, indicating robust growth and well-maintained condition. It usually flourishes near water bodies and trees, thriving in well-watered and sunny areas.',
            'A sample of Stressed grass. Stressed grass looks pale or faded due to lack of water or heavy foot traffic. It is commonly found near residential or commercial areas with frequent human activity.',
            'A sample of Synthetic grass. Synthetic grass always looks the same because it is made from artificial materials. It is widely used in sports facilities and artificial landscapes as a low-maintenance alternative to natural grass.',
            'A sample of Synthetic Tree. Trees have a strong trunk and a wide canopy of leaves. They are frequently seen in green spaces, parks, residential areas, and along streets, offering shade and beautifying the urban environment.',
            'A sample of Soil. Soil forms the surface of the ground, providing a foundation for the growth of vegetation. Its characteristics, including texture and composition, may vary throughout the image.',
            'A sample of Water. Water denotes various water bodies like lakes, rivers, and ponds. It is frequently bordered by healthy grass and trees, serving as a vital water source for vegetation.',
            'A sample of Residential. Residential areas consist of housing structures and are typically situated near commercial areas and road networks.',
            'A sample of Commercial. Commercial areas are characterized by various business activities and are commonly located along roads and highways.',
            'A sample of Road. Roads traverse the image, connecting different areas and serving as a vital transportation network.',
            'A sample of Highway. Highways are major arterial roads designed for high-volume traffic and long-distance travel.',
            'A sample of Railway. Railways are transportation routes for trains and may run parallel to roads or cross the image, connecting urban areas.',
            'A sample of Parking lot 1. Parking lot 1 represents areas designated for vehicle parking. It may be smaller and commonly found near residential or commercial zones',
            'A sample of Parking lot 2. Parking lot 2 represents areas designated for vehicle parking. It may be larger and situated near commercial areas or sports facilities.',
            'A sample of Tennis court. Tennis courts are specialized sports facilities designed exclusively for tennis activities.',
            'A sample of Running track. Running tracks are circular or oval tracks used for athletics and may be located in sports complexes or public grounds.',
            ]
    elif "Augsburg" in config.DATA.DATASET_NAME:
        text = [
            'A sample of Forest. Typically green with complex textures from dense trees and vegetation, often bordering Low Plants and Water areas.',
            'A sample of Residential Area. Features various building colors and regular textures from houses and roads, adjacent to Commercial Areas and near Allotments.',
            'A sample of Industrial Area. Commonly gray or brown with hard textures from factories and machinery, similar to Commercial Areas but separated from Residential Areas.',
            'A sample of Low Plants. Primarily green with simple textures from grasslands and low shrubs, sharing natural environments with Forests and Water areas.',
            'A sample of Allotment. Varied colors with textures from soil, plants, and small structures, typically located within or near Residential Areas.',
            'A sample of Commercial Area. Diverse buildings with bright colors and varied textures like signs and glass facades, closely linked to Residential and Industrial Areas.',
            'A sample of Water. Typically blue or green with wave-like textures, often bordering Forests, Low Plants, and near Residential and Commercial Areas.',
            ]
    else:
        raise Exception("main_RS.py is Wrong. Line 107")
    text_net = Text_Net(embed_dim=512,
                        context_length=77,
                        vocab_size=49408,
                        transformer_width=512,
                        transformer_heads=8,
                        transformer_layers=12)
    text_net.load_state_dict(torch.load("./text_encoder.pth"))
    text_net = text_net.cuda()
    text = clip.tokenize(text).cuda()
    feature_text = text_net(text)
    return feature_text

def cal_loss(criterion, outputs_1, labels, Z_text, outputs_2=None, is_pretrain=True, pred=None, loss_list=None):
    Z1 = outputs_1[-1]
    if is_pretrain:
        capacity = len(labels) // config.DATA.NUM_CLASSES
        Queue_list = [Queue(capacity=capacity, dim=Z1.shape[-1]) for _ in range(config.DATA.NUM_CLASSES)]
        for n, label in enumerate(labels):
            Queue_list[label].enqueue(Z1[n, :])
        loss_clip_all = 0
        for n in range(capacity):
            temp = [torch.unsqueeze(queue.dequeue(), dim=0).cuda() for queue in Queue_list]
            loss_clip_all += criterion["clip_loss"](image_features=torch.cat(temp, dim=0), text_features=Z_text)  # ClipLoss
        return (loss_clip_all / capacity)
    else:
        Z2 = outputs_2[-1]
        capacity = len(labels) // config.DATA.NUM_CLASSES
        Queue_list_teacher = [Queue(capacity=capacity, dim=Z1.shape[-1]) for _ in range(config.DATA.NUM_CLASSES)]
        for n, label in enumerate(labels):
            Queue_list_teacher[label].enqueue(Z1[n, :])
        Queue_list_student = [Queue(capacity=capacity, dim=Z2.shape[-1]) for _ in range(config.DATA.NUM_CLASSES)]
        for n, label in enumerate(labels):
            Queue_list_student[label].enqueue(Z2[n, :])
        loss_clip_all = 0
        loss_class_relation_kd = 0
        for n in range(capacity):
            temp_teacher = [torch.unsqueeze(queue.dequeue(), dim=0).cuda() for queue in Queue_list_teacher]
            temp_student = [torch.unsqueeze(queue.dequeue(), dim=0).cuda() for queue in Queue_list_student]
            loss_clip_all += criterion["clip_loss"](image_features=torch.cat(temp_student, dim=0), text_features=Z_text)  # ClipLoss
            logits_T = criterion["clip_loss"](image_features=torch.cat(temp_teacher, dim=0), text_features=Z_text, cal_logits=True)
            logits_S = criterion["clip_loss"](image_features=torch.cat(temp_student, dim=0), text_features=Z_text, cal_logits=True)
            for i, (logit_T, logit_S) in enumerate(zip(logits_T, logits_S)):
                loss_class_relation_kd += criterion["dist_loss"](logit_S, logit_T)
        loss_clip_all = loss_clip_all  / capacity
        loss_class_relation_kd = loss_class_relation_kd  / capacity
        loss_features_kd = 0
        for i in range(len(outputs_2)):
            loss_features_kd += criterion["mse_loss"](outputs_1[i], outputs_2[i])
        loss_features_kd /= len(outputs_2)
        loss =  loss_clip_all  + loss_class_relation_kd + loss_features_kd
        return loss, loss_clip_all, loss_features_kd, loss_class_relation_kd


def pre_train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, loss_scaler=None, is_pretrain=True, Z_text=None, lr_scheduler=None):
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    train_OA_meter = AverageMeter()
    model.train()
    start = time.time()
    end = time.time()
    optimizer.zero_grad()
    for idx, (data, targets) in enumerate(data_loader):
        for i in range(len(data)):
            data[i] = data[i].cuda()
        targets = targets.cuda()
        outputs, loss_list = model(x=data, mask=None)
        loss = cal_loss(criterion, outputs, targets, is_pretrain=is_pretrain, Z_text=Z_text, loss_list = loss_list) / config.TRAIN.ACCUMULATION_STEPS
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        pred = torch.nn.functional.softmax(criterion["clip_loss"](outputs[-1], Z_text, cal_logits=True)[0], dim=1).argmax(-1)
        train_OA = sum(pred == targets) / len(targets)
        loss_meter.update(loss.item(), targets.size(0))
        batch_time.update(time.time() - end)
        train_OA_meter.update(train_OA.cpu().item())
        end = time.time()
        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f'Train: [{epoch + 1}/{config.TRAIN.EPOCHS}][{idx + 1}/{num_steps}] || lr {lr:.6f} || '
            f'time {batch_time.val:.4f} ({batch_time.avg:.4f}) || '
            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) || '
            f'train_OA {train_OA_meter.val:.4f} ({train_OA_meter.avg:.4f})')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time))}")

def train_one_epoch_fine_tuning(config, teacher_model, student_model, criterion, data_loader, optimizer, epoch, loss_scaler=None, is_pretrain=True, Z_text=None, lr_scheduler=None,missing_modality_num=None):

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_clip_all_meter = AverageMeter()
    loss_feature_meter = AverageMeter()
    loss_class_relation_meter = AverageMeter()
    train_OA_meter = AverageMeter()
    teacher_model.eval()
    student_model.train()
    optimizer.zero_grad()
    for idx, (data, targets) in enumerate(data_loader):
        for i in range(len(data)):
            data[i] = data[i].cuda()
        targets = targets.cuda()

        with torch.no_grad():
            cloned_data_T = [tensor.clone() for tensor in data]
            outputs_T, _ = teacher_model(x=cloned_data_T, mask=None)

        if missing_modality_num is not None and set(missing_modality_num).issubset(set([num for num in range(config.DATA.Modality_Num)])): # 分别训练缺失1和缺失0
            mask = torch.ones(size=(len(data[0]), config.DATA.Modality_Num)).cuda()
            for i in missing_modality_num:
                mask[:, i] = 0
        else:
            mask = get_approx_uniform_masks(B=len(data[0]), N=len(data)).cuda()

        cloned_data_S = [tensor.clone() for tensor in data]
        outputs_S, loss_list = student_model(x=cloned_data_S, mask=mask)
        loss_tuple  = cal_loss(criterion=criterion, outputs_1=outputs_T, outputs_2=outputs_S, labels=targets,
                               is_pretrain=is_pretrain, Z_text=Z_text, loss_list=loss_list)
        loss, loss_clip_all, loss_feature, loss_class_relation = tuple(val / config.Fine_TUNE.ACCUMULATION_STEPS for val in loss_tuple)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=config.Fine_TUNE.CLIP_GRAD,
                                parameters=student_model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.Fine_TUNE.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.Fine_TUNE.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.Fine_TUNE.ACCUMULATION_STEPS)

        pred = torch.nn.functional.softmax(criterion["clip_loss"](outputs_S[-1], Z_text, cal_logits=True)[0], dim=1).argmax(-1)
        train_OA = sum(pred == targets) / len(targets)
        loss_meter.update(loss.item(), targets.size(0))
        loss_clip_all_meter.update(loss_clip_all.item(), targets.size(0))
        loss_feature_meter.update(loss_feature.item(), targets.size(0))
        loss_class_relation_meter.update(loss_class_relation.item(), targets.size(0))
        train_OA_meter.update(train_OA.cpu().item())
        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f'Train: [{epoch + 1}/{config.Fine_TUNE.EPOCHS}][{idx + 1}/{num_steps}] || lr {lr:.6f} || '
            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) || '
            f'loss_clip_all {loss_clip_all_meter.val:.4f} ({loss_clip_all_meter.avg:.4f}) || '
            f'loss_feature {loss_feature_meter.val:.4f} ({loss_feature_meter.avg:.4f}) || '
            f'loss_class_relation {loss_class_relation_meter.val:.4f} ({loss_class_relation_meter.avg:.4f}) || '
            f'train_OA {train_OA_meter.val:.4f} ({train_OA_meter.avg:.4f})')


@torch.no_grad()
def test(config, data_loader, model, Z_text, gt, OA_best, fine_tune=False, missing_modality_num=[], criterion=None):
    model.eval()
    pred_map = torch.zeros((gt.shape[0], gt.shape[-1]))
    pred_temp = []
    for idx, (images, target) in enumerate(data_loader):
        for i in range(len(images)):
            images[i] = images[i].cuda()

        if fine_tune:
            if set(missing_modality_num).issubset(set([num for num in range(config.DATA.Modality_Num)])):
                mask = torch.ones(size=(len(images[0]), config.DATA.Modality_Num)).cuda()
                for i in missing_modality_num:
                    mask[:, i] = 0
                    images[i].fill_(0)
            outout_features, _ = model(x=images, mask=mask)
        else:
            outout_features, _  = model(x=images, mask=None)

        pred_temp.extend(torch.nn.functional.softmax(criterion["clip_loss"](outout_features[-1], Z_text, cal_logits=True)[0], dim=1).argmax(-1).cpu().numpy() + 1)

    idx, idy = np.where(gt!=0)
    for i, (x, y) in enumerate(zip(idx, idy)):
        pred_map[x, y] = pred_temp[i]

    ## classfication report
    test_pred = pred_map[gt != 0]
    test_true = gt[gt != 0]
    OA = accuracy_score(test_true, test_pred)
    AA = recall_score(test_true, test_pred, average='macro')
    kappa = cohen_kappa_score(test_true, test_pred)
    if "Houston" in config.DATA.DATASET_NAME:
        class_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']  # Houston
    elif "Trento" in config.DATA.DATASET_NAME:
        class_name = ['1', '2', '3', '4', '5', '6']  # Trento
    elif "Augsburg" in config.DATA.DATASET_NAME:
        class_name = ['1', '2', '3', '4', '5', '6', '7']  # Augsburg
    report_log = classification_report(test_true, test_pred, target_names=class_name, digits=4)
    logger.info(report_log)
    logger.info(f"OA: {OA}, AA: {AA}, kappa: {kappa}")
    if (OA_best < OA):
        OA_best = OA
        if not os.path.exists(f"{config.OUTPUT}/model"):
            os.makedirs(f"{config.OUTPUT}/model")
        if not os.path.exists(f'{config.OUTPUT}/res'):
            os.makedirs(f'{config.OUTPUT}/res')
    torch.save(model.state_dict(), f"{config.OUTPUT}/model/{config.DATA.DATASET_NAME}_{OA:.4f}.pth")
    sio.savemat(f'{config.OUTPUT}/res/predMap_{config.DATA.DATASET_NAME}_{OA:.4f}.mat', {'data': pred_map.numpy()})
    return OA, OA_best


def main(is_pretrain=True):
    data_loader_train, data_loader_test, gt = build_loader()
    loss_scaler = NativeScalerWithGradNormCount()
    Z_text = get_text_feature()
    criterion = {
        "clip_loss": ClipLoss(),
        "mse_loss": torch.nn.MSELoss(),
        "dist_loss": DIST(),
        "cross_entropy_loss": torch.nn.CrossEntropyLoss(),
        "Kl_loss": DistillKL(T=4),
    }

    if is_pretrain:
        logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_model(config, is_pretrain=True)
        model = model.cuda()

        logger.info(f"所有参数为：{sum(p.numel() for p in model.parameters()) / 1e3} K || {sum(p.numel() for p in model.parameters()) / 1e6} M")
        logger.info( f"可训练参数为：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3} K || {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6} M")

        optimizer = build_optimizer(config, model, logger)
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
        else:
            lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
        logger.info("Start Pre-Training")
        OA_1 = 0
        OA_best_complete = 0
        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS): #(0, 300)
            pre_train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, loss_scaler, is_pretrain, Z_text, lr_scheduler)
            if (epoch + 1) % 50 == 0:
                OA_1, OA_best_complete = test(config, data_loader_test, model, Z_text, gt, OA_best_complete, fine_tune=False, criterion=criterion)
                logger.info(f"Test -- Complete1 ================== epoch: {epoch + 1} || OA : {OA_1 :.4f} || OA_best : {OA_best_complete :.4f}")
        logger.info(f"Output Folder : {config.OUTPUT}")

    else:
        logger.info(f"Creating Teacher and Student Model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        teacher_model = build_model(config, is_pretrain=True)
        student_model = build_model(config, is_pretrain=False)

        teacher_model.load_state_dict(torch.load(config.MODEL_PATH), strict=True)
        student_model.load_state_dict(torch.load(config.MODEL_PATH), strict=False)

        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()

        logger.info(f"Total number of parameters: {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad) / 1e3} K || {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad) / 1e6} M")
        logger.info(f"Total number of parameters: {sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e3} K || {sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e6} M")

        for name, param in teacher_model.named_parameters(recurse=True):
            param.requires_grad = False

        logger.info(f"After Freezing - Total number of parameters : {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad) / 1e3} K || {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad) / 1e6} M")
        logger.info(f"After Freezing - Total number of parameters: {sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e3} K || {sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e6} M")

        optimizer = build_optimizer_fine_tuning(config, student_model, logger)
        if config.Fine_TUNE.ACCUMULATION_STEPS > 1:
            lr_scheduler = build_scheduler_fine_tuning(config, optimizer, len(data_loader_train) // config.Fine_TUNE.ACCUMULATION_STEPS)
        else:
            lr_scheduler = build_scheduler_fine_tuning(config, optimizer, len(data_loader_train))
        logger.info("Start Training")
        OA_best_complete = 0
        OA_best_0 = 0
        OA_best_1 = 0
        ovrall_OA_best = 0
        for epoch in range(config.Fine_TUNE.START_EPOCH, config.Fine_TUNE.EPOCHS):

            train_one_epoch_fine_tuning(config, teacher_model, student_model, criterion, data_loader_train, optimizer, epoch, loss_scaler,
                                              is_pretrain, Z_text, lr_scheduler, missing_modality_num=None)

            if (epoch + 1) % 50 == 0:
                OA_all, OA_best_complete = test(config, data_loader_test, student_model, Z_text, gt, OA_best_complete, fine_tune=False, criterion=criterion)
                logger.info(f"Test -- Complete2 ================== epoch: {epoch + 1} || OA : {OA_all :.4f} || OA_best : {OA_best_complete :.4f}")
                OA_00, OA_best_0 = test(config, data_loader_test, student_model, Z_text, gt, OA_best_0, fine_tune=True, missing_modality_num=[0], criterion=criterion)
                logger.info(f"Test -- Missing(0)-Only(1) ================== epoch: {epoch + 1 } || OA : {OA_00 :.4f} || OA_best : {OA_best_0 :.4f}")
                OA_11, OA_best_1 = test(config, data_loader_test, student_model, Z_text, gt, OA_best_1, fine_tune=True, missing_modality_num=[1], criterion=criterion)
                logger.info(f"Test -- Missing(1)-Only(0) ================== epoch: {epoch + 1 } || OA : {OA_11 :.4f} || OA_best : {OA_best_1 :.4f}")

                ovrall_OA = OA_all + OA_00 + OA_11
                if (ovrall_OA_best < ovrall_OA):
                    ovrall_OA_best = ovrall_OA
                    if not os.path.exists(f"{config.OUTPUT}/model"):
                        os.makedirs(f"{config.OUTPUT}/model")
                    if not os.path.exists(f'{config.OUTPUT}/res'):
                        os.makedirs(f'{config.OUTPUT}/res')
                    torch.save(student_model.state_dict(), f"{config.OUTPUT}/model/{config.DATA.DATASET_NAME}_{OA_all:.4f}_{OA_00:.4f}_{OA_11:.4f}.pth")

        if not os.path.exists(f"{config.OUTPUT}/model"):
            os.makedirs(f"{config.OUTPUT}/model")
        torch.save(student_model.state_dict(), f"{config.OUTPUT}/model/{config.DATA.DATASET_NAME}_last.pth")
        logger.info(f"Output Folder : {config.OUTPUT}")


if __name__ == '__main__':
    import os

    os.environ["RANK"] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(config.MASTER_PORT_)

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if True:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.Is_Pretrain = args.Is_Pretrain
    config.MODEL_PATH = args.MODEL_PATH
    config.freeze()

    config.defrost()
    if dist.get_rank() == 0:
        obj = [config.OUTPUT]
    else:
        obj = [None]
    dist.broadcast_object_list(obj)
    dist.barrier()
    config.OUTPUT = obj[0]
    print(config.OUTPUT, flush=True)
    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    if args.memory_limit_rate > 0 and args.memory_limit_rate < 1:
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_rate)
        usable_memory = torch.cuda.get_device_properties(0).total_memory * args.memory_limit_rate / 1e6
        print(f"===========> GPU memory is limited to {usable_memory}MB", flush=True)

    main(is_pretrain=config.Is_Pretrain)
