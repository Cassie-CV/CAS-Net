'''
@author: caixia_dong
@license: (C) Copyright 2020-2023, Medical Artificial Intelligence, XJTU.
@contact: caixia_dong@xjtu.edu.cn
@software: MedAI
@file: train.py
@time: 2022/7/22 14:49
@version:
@desc:
'''
import torch.nn as nn

from model.csnet_3d import CSNet3D
from model.unet3d import UNet3D

from dataloader.npy_3d_Loader import *
import pandas as pd

from postprocess.keep_the_largest_area import get_aorta_branch
from postprocess.keep_the_largest_area import backpreprcess as postprocess
from postprocess.get_patch import get_patch_new
from utils.evaluation_metrics3D import metrics_3d, Dice, over_rate, under_rate

Test_Model = {'CSNet3D': CSNet3D,
              'UNet3D': UNet3D
              }

# os.environ['CUDA_VISIBLE_DEVICES'] = "5"

args = {
    'root': 'cta_project/code/CAS-Net',
    'data_path': '/cta_project/data/npy',
    'pred_path': 'cor_result_160_CASNet',
    'input_shape': (128, 160, 160),
    'model_path': './save_models_randomcrop',
    'batch_size': 2,
    'folder': 'folder1',
    'model_name': 'CSNet3D',
}

if not os.path.exists(args['pred_path']):
    os.makedirs(args['pred_path'])
save_path = os.path.join(args['pred_path'], args['model_name'] + '_' + args['folder'])

if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path_label = os.path.join(save_path, 'label')
save_path_pred = os.path.join(save_path, 'pred')
if not os.path.exists(save_path_label):
    os.makedirs(save_path_label)
if not os.path.exists(save_path_pred):
    os.makedirs(save_path_pred)


def load_net():
    model = Test_Model[args['model_name']](2, 1).cuda()
    ckpt_path = os.path.join(args['model_path'], args['model_name'] + '_' + args['folder'])
    modelname = ckpt_path + '/' + 'best_score' + '_checkpoint.pkl'

    # net.load_state_dict(checkpoint)
    try:  # single GPU model_file
        # model.load_state_dict(torch.load(model_file), strict=True)
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint)
    except:  # multi GPU model_file
        # state_dict = torch.load(model_file)

        net = nn.DataParallel(model)
        torch.save(net.module.state_dict(), "model.pth")
        pretrained_dict = torch.load("model.pth")
        model.load_state_dict(pretrained_dict)
    return model


def get_prediction(pred):
    pred = torch.argmax(pred, dim=1)
    mask = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    # print(np.max(mask),np.min(mask))
    mask = mask.squeeze(0)  # for CE Loss
    return mask


def get_metrics(pred, gt):
    pred[pred > 0] = 255
    gt[gt > 0] = 255
    Ur = under_rate(pred, gt)
    Or = over_rate(pred, gt)
    dice = Dice(pred, gt)
    tp, fn, fp, IoU = metrics_3d(pred, gt)
    return tp, fn, fp, IoU, dice, Or, Ur


def model_eval(net):
    print("\033[1;30;43m {} Start testing ... {}\033[0m".format("*" * 8, "*" * 8))
    images_lsts, groundtruth_lsts = load_dataset(args['data_path'], args['folder'], False)
    patch_size1 = args['input_shape'][0]
    patch_size2 = args['input_shape'][1]
    TP, FN, FP, IoU, Dice, OR, UR = [], [], [], [], [], [], []
    data_arry = []
    data_arrya = []
    data_arryb = []
    file_num = 0
    with torch.no_grad():
        net.eval()
        for idx in range(len(images_lsts)):
            # try:
            img_path = images_lsts[idx]
            filename = os.path.basename(img_path)[:-4]
            gt_path = groundtruth_lsts[idx]

            image = np.load(img_path)  # sitk.ReadImage(img_path)
            label_npy = np.load(gt_path)
            label_npy[label_npy > 0] = 1
            mask = np.zeros_like(image).astype(np.uint8)

            patch_roi_list = get_patch_new(image.shape, patch_size1, patch_size2, 4)
            for j, patch_roi in enumerate(patch_roi_list):
                zz1, xx1, yy1 = patch_roi
                # k=k+1
                if zz1 == -1:
                    flag = 1
                    continue
                img_patch1 = image[zz1:zz1 + patch_size1, xx1:xx1 + patch_size2, yy1:yy1 + patch_size2].astype(
                    np.float32)
                img_patch = torch.from_numpy(np.ascontiguousarray(img_patch1)).unsqueeze(0).unsqueeze(0)
                img_patch = img_patch / 255
                img_patch = img_patch.cuda()
                mask_temp = np.zeros_like(image).astype(np.uint8)
                output = net(img_patch)
                # print(np.max(output))
                mask_patch = get_prediction(output)
                # print(np.max(mask_patch))
                mask_temp[zz1:zz1 + patch_size1, xx1:xx1 + patch_size2, yy1:yy1 + patch_size2] = mask_patch
                mask_temp[mask_temp > 0] = 1
                mask = mask | mask_temp
            print(np.max(mask))

            mask = postprocess(mask)
            label_npy = np.flipud(label_npy)
            mask = np.flipud(mask)
            print('+++++++++++++')
            mask_aorta, mask_branch = get_aorta_branch(mask)
            label_aorta, label_branch = get_aorta_branch(label_npy)
            print('+++++++++++++')
            tp, fn, fp, iou, dice, Or, Ur = get_metrics(mask, label_npy)
            tpa, fna, fpa, ioua, dicea, Ora, Ura = get_metrics(mask_aorta, label_aorta)
            tpb, fnb, fpb, ioub, diceb, Orb, Urb = get_metrics(mask_branch, label_branch)
            print(
                "--- test TP:{0:.4f}    test FN:{1:.4f}    test FP:{2:.4f}    test IoU:{3:.4f}  test Dice:{4:.4f} test OR:{5:.4f}  test UR:{6:.4f}".format(
                    tp, fn, fp, iou, dice, Or, Ur))
            print(
                "--- test TPa:{0:.4f}    test FNa:{1:.4f}    test FPa:{2:.4f}    test IoUa:{3:.4f}  test Dicea:{4:.4f} testa OR:{5:.4f}  test UR:{6:.4f}".format(
                    tpa, fna, fpa, ioua, dicea, Ora, Ura))
            print(
                "--- test TP:{0:.4f}    test FN:{1:.4f}    test FP:{2:.4f}    test IoU:{3:.4f}  test Dice:{4:.4f} test OR:{5:.4f}  test UR:{6:.4f}".format(
                    tpb, fnb, fpb, ioub, diceb, Orb, Urb))
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
            IoU.append(iou)
            Dice.append(dice)
            OR.append(Or)
            UR.append(Ur)
            data_arry.append([filename, tp, fn, fp, iou, dice, Or, Ur])
            data_arrya.append([filename, tpa, fna, fpa, ioua, dicea, Ora, Ura])
            data_arryb.append([filename, tpb, fnb, fpb, ioub, diceb, Orb, Urb])
            mask[mask > 0] = 1
            label_npy[label_npy > 0] = 1
            out = sitk.GetImageFromArray(mask)
            sitk.WriteImage(out, os.path.join(save_path_pred, filename + '.nii.gz'))
            out1 = sitk.GetImageFromArray(label_npy * 2)
            sitk.WriteImage(out1, os.path.join(save_path_label, filename + '.nii.gz'))
            print("save_ok!!!!")

    save = pd.DataFrame(data_arry, columns=['patient_name', 'TP', 'FN', 'FP', 'IoU', 'Dice', 'UR', 'OR'])
    save.to_csv(os.path.join(save_path, 'test_result.csv'), index=False, header=True)
    savea = pd.DataFrame(data_arrya, columns=['patient_name', 'TP', 'FN', 'FP', 'IoU', 'Dice', 'UR', 'OR'])
    savea.to_csv(os.path.join(save_path, 'test_result_a.csv'), index=False, header=True)
    saveb = pd.DataFrame(data_arryb, columns=['patient_name', 'TP', 'FN', 'FP', 'IoU', 'Dice', 'UR', 'OR'])
    saveb.to_csv(os.path.join(save_path, 'test_result_b.csv'), index=False, header=True)
    print("TP,FN,FP,IoU,Dice,UR,OR", np.mean(TP), np.mean(FN), np.mean(FP), np.mean(IoU), np.mean(Dice), np.mean(OR),
          np.mean(UR))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = load_net()
    model_eval(net)
    # predict_whole()
