import copy
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from mmcv import imwrite
from mmpose.core import transform_preds
from mmpose.core.evaluation import keypoint_pck_accuracy
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
from mmcv.runner.base_module import BaseModule
from mmpose.models import HEADS, TopdownHeatmapBaseHead, build_loss
from scape.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence,
                                         build_positional_encoding)
from scape.models.utils import build_transformer
from torch import nn
from timm.models.layers.weight_init import trunc_normal_
from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.models.utils.ops import resize
import math
from mmcv.cnn import (Conv2d, Linear, xavier_init, build_upsample_layer, ConvModule,
                      constant_init, normal_init, build_conv_layer, build_norm_layer)

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1




MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1



def clone_module(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def show_result(
            img,
            result,
            pck=0,
            skeleton=None,
            kpt_score_thr=0.3,
            bbox_color='green',
            pose_kpt_color_list=None,
            pose_limb_color=None,
            radius=4,
            text_color=(255, 0, 0),
            thickness=1,
            font_scale=0.5,
            win_name='',
            show=False,
            out_dir_lei='',
            wait_time=0,
            mask=None,
            out_file_=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_limb_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img_path = img
        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape
        mask = mask
        bbox_result = []
        pose_result = []
        result[0]['keypoints'] = result[0]['keypoints']['preds'][:, :, :2]
        result[1]['keypoints'] = (result[1]['keypoints']['preds'][:, :, :2]) * np.repeat(mask.cpu().detach().numpy(), 2,
                                                                                         -1)
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            mmcv.imshow_bboxes(
                img,
                bboxes,
                colors=bbox_color,
                top_k=-1,
                thickness=thickness,
                show=False,
                win_name=win_name,
                wait_time=wait_time,
                out_file=None)

            for person_id, kpts in enumerate(pose_result):
                # draw each point on image
                kpts = kpts[0]
                pose_kpt_color = pose_kpt_color_list[person_id]
                if pose_kpt_color is not None:
                    assert len(pose_kpt_color) == len(kpts)
                    for kid, kpt in enumerate(kpts):
                        x_coord, y_coord = int(kpt[0]), int(kpt[1])
                        # if kpt_score > kpt_score_thr:
                        if 1 == 1:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            font = cv2.FONT_HERSHEY_SIMPLEX  # ��������
                            cv2.putText(img_copy, str(kid), (int(x_coord), int(y_coord)), font, 0.8, (255, 255, 255), 2)
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            # transparency = max(0, min(1, kpt_score))
                            transparency = 0.5
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)
                    out_file = None
                    # out_file = '/data/yjliang/code/Category-Agnostic-Pose-Estimation/P2/Pose-for-Everything/work_dirs/token_3_bs16_spilt1_dri_no_jian_no_pos/out_img/' + out_dir_lei + '/'
                    # if not os.path.exists(out_file):  # ���·��������
                    #    os.makedirs(out_file)
                    if out_file is not None:
                        imwrite(img, out_file + str(person_id) + '_' + str(pck) + out_file_)

                # draw limbs
                if skeleton is not None and pose_limb_color is not None:
                    assert len(pose_limb_color) == len(skeleton)
                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                                  1]))
                        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                                  1]))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1, 2] > kpt_score_thr
                                and kpts[sk[1] - 1, 2] > kpt_score_thr):
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle),
                                0, 360, 1)

                            r, g, b = pose_limb_color[sk_id]
                            cv2.fillConvexPoly(img_copy, polygon,
                                               (int(r), int(g), int(b)))
                            transparency = max(
                                0,
                                min(
                                    1, 0.5 *
                                       (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

        show, wait_time = 1, 1
        if show:
            height, width = img.shape[:2]
            max_ = max(height, width)

            factor = min(1, 800 / max_)
            enlarge = cv2.resize(
                img, (0, 0),
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_CUBIC)
        plt.axis('off')
        plt.imshow(img)
        # plt.show()
        out_file = None

        if out_file is not None:
            imwrite(img, out_file + img_path.split('/')[-1])

        return img

def _get_3rd_point(a, b):
        """To calculate the affine matrix, three pairs of points are required. This
        function is used to get the 3rd point, given 2D points a & b.

        The 3rd point is defined by rotating vector `a - b` by 90 degrees
        anticlockwise, using b as the rotation center.

        Args:
            a (np.ndarray): point(x,y)
            b (np.ndarray): point(x,y)

        Returns:
            np.ndarray: The 3rd point.
        """
        assert len(a) == 2
        assert len(b) == 2
        direction = a - b
        third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

        return third_pt

def rotate_point(pt, angle_rad):
        """Rotate a point by an angle.

        Args:
            pt (list[float]): 2 dimensional point to be rotated
            angle_rad (float): rotation angle by radian

        Returns:
            list[float]: Rotated point.
        """
        assert len(pt) == 2
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        new_x = pt[0] * cs - pt[1] * sn
        new_y = pt[0] * sn + pt[1] * cs
        rotated_pt = [new_x, new_y]

        return rotated_pt

def get_affine_transform(center,
                             scale,
                             rot,
                             output_size,
                             shift=(0., 0.),
                             inv=False):
        """Get the affine transform matrix, given the center/scale/rot/output_size.

        Args:
            center (np.ndarray[2, ]): Center of the bounding box (x, y).
            scale (np.ndarray[2, ]): Scale of the bounding box
                wrt [width, height].
            rot (float): Rotation angle (degree).
            output_size (np.ndarray[2, ]): Size of the destination heatmaps.
            shift (0-100%): Shift translation ratio wrt the width/height.
                Default (0., 0.).
            inv (bool): Option to inverse the affine transform direction.
                (inv=False: src->dst or inv=True: dst->src)

        Returns:
            np.ndarray: The transform matrix.
        """
        assert len(center) == 2
        assert len(scale) == 2
        assert len(output_size) == 2
        assert len(shift) == 2

        # pixel_std is 200.
        scale_tmp = scale * 200.0

        shift = np.array(shift)
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = rotate_point([0., src_w * -0.5], rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        src[2, :] = _get_3rd_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans


def vis_support(xx1):
    plt_list = []
    plt_list1 = []
    for jjj in range(1):
        x1 = xx1[0, :100, :].reshape(xx1.shape[0], 100, 16, 16)
        for i in range(len(img_metas)):
            file_path = img_metas[i]['sample_image_file'][0]
            c = img_metas[i]['sample_center'][0]
            s = img_metas[i]['sample_scale'][0]
            r = img_metas[i]['sample_rotation'][0]

            data_numpy2 = cv2.imread(file_path, cv2.IMREAD_COLOR)

            trans = get_affine_transform(c, s, r, [256, 256])
            data_numpy2 = cv2.warpAffine(
                data_numpy2,
                trans, (256, 256),
                flags=cv2.INTER_LINEAR)

            data_numpy2 = cv2.resize(data_numpy2, (64, 64), 0, 0)
            x1 = F.interpolate(x1, size=(64, 64), mode='bilinear', align_corners=False)
            for j in range(100):
                # if mask_ss[i][j][0] == 0:
                if j != 13:
                    continue

                target_s_heat = np.uint8((target_sss[i][j] * 255).cpu().detach().numpy())
                target_q_heat = np.uint8((qurey_sss[i][j] * 255).cpu().detach().numpy())
                x1[i][j] = x1[i][j] * (1 / x1[i][j].max())
                s_heat1 = np.uint8((x1[i][j] * 255).cpu().detach().numpy())

                # s_heat1 = s_heat1.resize(64, 64)

                hit_img_t = cv2.cvtColor(target_s_heat, cv2.COLOR_RGB2BGR)
                # hit_img_t=cv2.cvtColor(np.asarray(target_s[i][j].cpu().detach().numpy()), cv2.COLOR_RGB2BGR)
                hit_img1 = cv2.applyColorMap(s_heat1, cv2.COLORMAP_HSV)  # Image��ʽת����cv2��ʽ
                hit_img_q = cv2.applyColorMap(target_q_heat, cv2.COLORMAP_HSV)
                alpha = 0.5  # ���ø���ͼƬ��͸����
                # cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (255, 0, 0), -1)  # ������ɫΪ�ȶ�ͼ����ɫ��ɫ
                # image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)  # �������ȶ�ͼ���ǵ�ԭͼ

                image2 = cv2.addWeighted(hit_img1, alpha, data_numpy2, 1 - alpha, 0)
                image2 = cv2.addWeighted(hit_img1, alpha, data_numpy2, 1 - alpha, 0)  # ���ȶ�ͼ���ǵ�ԭͼ
                image3 = cv2.addWeighted(hit_img_q, alpha, data_numpy2, 1 - alpha, 0)
                if jjj == 0 and j == 13:
                    plt_list.append(image3)
                if j == 13:
                    plt_list1.append(image2)
    plt.figure()
    plt.axis('off')  # ����ʾ������
    for i in range(1):
        plt.subplot(1, 1, i + 1)
        plt.axis('off')  # ����ʾ������
        plt.imshow(plt_list[i])
    plt.show()
    plt.clf()

    plt.figure()
    plt.axis('off')  # ����ʾ������
    for i in range(1):
        plt.subplot(1, 1, i + 1)
        plt.axis('off')  # ����ʾ������
        plt.imshow(plt_list1[i])
    plt.show()



def vis_query(xx1):
        plt_list = []
        plt_list1 = []
        for jjj in range(1):
            x1 = xx1[0, :100, :].reshape(xx1.shape[0], 100, 16, 16)
            for i in range(len(img_metas)):
                file_path = img_metas[i]['query_image_file']
                c = img_metas[i]['query_center']
                s = img_metas[i]['query_scale']
                r = img_metas[i]['query_rotation']

                data_numpy2 = cv2.imread(file_path, cv2.IMREAD_COLOR)

                trans = get_affine_transform(c, s, r, [256, 256])
                data_numpy2 = cv2.warpAffine(
                    data_numpy2,
                    trans, (256, 256),
                    flags=cv2.INTER_LINEAR)

                data_numpy2 = cv2.resize(data_numpy2, (64, 64), 0, 0)
                x1 = F.interpolate(x1, size=(64, 64), mode='bilinear', align_corners=False)
                for j in range(100):
                    # if mask_ss[i][j][0] == 0:
                    if j != 13:
                        continue

                    target_s_heat = np.uint8((target_sss[i][j] * 255).cpu().detach().numpy())
                    target_q_heat = np.uint8((qurey_sss[i][j] * 255).cpu().detach().numpy())
                    x1[i][j] = x1[i][j] * (1 / x1[i][j].max())
                    s_heat1 = np.uint8((x1[i][j] * 255).cpu().detach().numpy())

                    # s_heat1 = s_heat1.resize(64, 64)

                    hit_img_t = cv2.cvtColor(target_s_heat, cv2.COLOR_RGB2BGR)
                    # hit_img_t=cv2.cvtColor(np.asarray(target_s[i][j].cpu().detach().numpy()), cv2.COLOR_RGB2BGR)
                    hit_img1 = cv2.applyColorMap(s_heat1, cv2.COLORMAP_HSV) # Image��ʽת����cv2��ʽ
                    hit_img_q = cv2.applyColorMap(target_q_heat, cv2.COLORMAP_HSV)
                    alpha = 0.4  # ���ø���ͼƬ��͸����
                    # cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (255, 0, 0), -1)  # ������ɫΪ�ȶ�ͼ����ɫ��ɫ
                    # image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)  # �������ȶ�ͼ���ǵ�ԭͼ


                    image2 = cv2.addWeighted(hit_img1, alpha, data_numpy2, 1 - alpha, 0)  # ���ȶ�ͼ���ǵ�ԭͼ
                    image3 = cv2.addWeighted(hit_img_q, alpha, data_numpy2, 1 - alpha, 0)
                    if jjj == 0 and j == 13:
                        plt_list.append(image3)
                    if j == 13:
                        plt_list1.append(image2)


        plt.figure()
        plt.axis('off')  # ����ʾ������
        for i in range(1):
            plt.subplot(1, 1, i + 1)
            plt.axis('off')  # ����ʾ������
            plt.imshow(plt_list[i])
        plt.show()
        plt.clf()

        plt.figure()
        plt.axis('off')  # ����ʾ������
        for i in range(1):
            plt.subplot(1, 1, i + 1)
            plt.axis('off')  # ����ʾ������
            plt.imshow(plt_list1[i])
        plt.show()

def inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

class Residual(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x, **kwargs):
            return self.fn(x, **kwargs) + x




class PreNorm(nn.Module):
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * fusion_factor)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x),  **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x,x2=None, **kwargs):
        return self.fn(x,x2, **kwargs) + x

class PreNorm2(nn.Module):
        def __init__(self, dim, fn, fusion_factor=1):
            super().__init__()
            self.norm = nn.LayerNorm(dim * fusion_factor)
            self.fn = fn

        def forward(self, x,x2=None, **kwargs):
            return self.fn(self.norm(x),x2, **kwargs)

class FeedForward2(nn.Module):
        def __init__(self, dim, hidden_dim, dropout=0.):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x,x2=None):
            return self.net(x)


class Attention2(nn.Module):
    def __init__(self, dim=256, heads=8, dropout=0., num_keypoints=100, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x,x2, mask=None):
        b, n, _, h = *x.shape, self.heads
        v = self.to_v(x2).reshape(x2.shape[0], x2.shape[1], h, -1).permute(0, 2, 1, 3)
        k= self.to_k(x2).reshape(x2.shape[0], x2.shape[1], h, -1).permute(0, 2, 1, 3)
        q= self.to_q(x).reshape(x.shape[0], x.shape[1], h, -1).permute(0, 2, 1, 3)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            # mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.unsqueeze(1).repeat(1, 100, 1)
            mask = torch.unsqueeze(mask, dim=1)
            mask = mask.repeat(1, 8, 1, 1)
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        # attn = attn

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Attention(nn.Module):
        def __init__(self, dim, heads=8, dropout=0., num_keypoints=None, scale_with_head=False):
            super().__init__()
            self.heads = heads
            self.expert = nn.Sequential(
                nn.ReLU(),
                nn.Linear(100, 50), nn.ReLU(),
                nn.Linear(50, 100), nn.ReLU())
            n_experts=4
            #self.experts = clone_module(self.mlp, n_experts)
            self.moes = clone_module(self.expert, n_experts)
            #self.c= Linear(
            #    64, 1)
            self.gates = nn.Sequential(nn.Dropout(0.6),nn.Linear(32, 4),nn.Dropout(0.2),nn.LayerNorm(4))
            self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5
            self.to_qk_q = nn.Linear(dim, dim * 2, bias=False)
            self.to_v = nn.Linear(dim, dim, bias=False)
            self.to_qk_s = nn.Linear(dim, dim * 2, bias=False)
            self.to_out = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Dropout(dropout)
            )
            self.num_keypoints = num_keypoints

        def forward(self, x, mask=None):
            b, n, _, h = *x.shape, self.heads

            #x_c=self.c(x[:,100:,:].detach().permute(0,2,1)).squeeze(2)

            qkv_s = self.to_qk_s(x[:, :100, :]).chunk(2, dim=-1)
            q_s, k_s = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv_s)
            v = self.to_v(x)
            v = v.reshape(v.shape[0], v.shape[1], h, -1).permute(0, 2, 1, 3)
            gates = self.gates(x.reshape(v.shape[0], v.shape[1],  v.shape[2],  v.shape[3]).detach()[:,:, :100, :]).softmax(-1)
            qkv_q = self.to_qk_q(x[:, 100:, :]).chunk(2, dim=-1)
            q_q, k_q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv_q)

            q = torch.cat((q_s, q_q), dim=2)
            v = v
            k = torch.cat((k_s, k_q), dim=2)
            dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
            mask_value = -torch.finfo(dots.dtype).max

            if mask is not None:
                # mask=torch.unsqueeze(x,dim=1)
                # mask = F.pad(mask.flatten(1), (1, 0), value = True)
                assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
                # mask = mask[:, None, :] * mask[:, :, None]
                mask = mask.unsqueeze(1).repeat(1, 100 + 64, 1)
                mask = torch.unsqueeze(mask, dim=1)
                mask = mask.repeat(1, 8, 1, 1)
                dots.masked_fill_(mask, mask_value)
                del mask
            attn_q = dots[:, :, :100, :100].contiguous()
            moes = [self.moes[i](attn_q)*gates[:,:,:,i,None]  for i in range(4)]   #16*8*100*100  scape: 16*8*100*5
            moes=sum(moes)
            kar = (dots[:, :, :100, :100].clone() + moes.contiguous())
            dots[:, :, :100, :100] = kar.clone().contiguous()
            attn = dots.softmax(dim=-1)

            out = torch.einsum('bhij,bhjd->bhid', attn, v)

            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            return out

class MLP(nn.Module):
        """ Very simple multi-layer perceptron (also called FFN)"""

        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.num_layers = num_layers
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            return x

@TRANSFORMER.register_module()
class Transformer(BaseModule):
        def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, all_attn=False,
                     scale_with_head=False):
            super().__init__()
            self.layers = nn.ModuleList([])
            self.all_attn = all_attn
            self.num_keypoints = num_keypoints
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, num_keypoints=num_keypoints,
                                                    scale_with_head=scale_with_head))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
                ]))

        def forward(self, x, mask=None,  pos=None):
            for idx, (attn, ff) in enumerate(self.layers):
                if idx > 0 and self.all_attn:
                    x[:, self.num_keypoints:] += pos
                x = attn(x,  mask=mask)
                x = ff(x)
            return x

class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, all_attn=False,
                 scale_with_head=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Attention2(dim, heads=heads, dropout=dropout, num_keypoints=num_keypoints,
                                                scale_with_head=scale_with_head))),
                Residual2(PreNorm2(dim, FeedForward2(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x,x2=None, mask=None):
        for idx, (attn, ff) in enumerate(self.layers):
            x = attn(x,x2, mask=mask)
            x = ff(x)
        return x

class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                                   bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                      momentum=BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

@HEADS.register_module()
class TokenPose_TB_base(TopdownHeatmapBaseHead):

        def __init__(self,
                     in_channels,
                     positional_encoding=dict(
                         type='SinePositionalEncoding',
                         num_feats=128,
                         normalize=True),
                     transformer=None,
                     loss_keypoint=None,
                     train_cfg=None,
                     test_cfg=None,
                     dim=256,
                     hidden_heatmap_dim=64 * 8,
                     heatmap_dim=64 * 64,
                     apply_multi=True,
                     apply_init=False,
                     emb_dropout=0,
                     num_keypoints=100,
                     heatmap_size=(64, 64)
                     ):
            # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
            # since it brings inconvenience when the initialization of
            # `AnchorFreeHead` is called.
            super().__init__()

            self.in_channels = in_channels
            self.heatmap_size = heatmap_size
            self.num_keypoints = num_keypoints
            self.positional_encoding = build_positional_encoding(positional_encoding)
            self.GKP=Decoder(256,2,8,256*2,0.5,100,False,True)
            self.transformer = build_transformer(transformer)
            self.pos_embedding = nn.Parameter(
                self._make_sine_position_embedding(256),
                requires_grad=False)
            self.dropout = nn.Dropout(emb_dropout)
            self.to_keypoint_token = nn.Identity()
            self.embed_dims = dim
            # self.mlp_head = nn.Sequential(
            #     nn.LayerNorm(dim),
            #     nn.Linear(dim, 2),
            # )
            self.mlp_head = MLP(dim, dim // 2, 2, 2)
            self.position = nn.Embedding(num_keypoints, 2)
            # self.mlp_head_list=nn.ModuleList(deepcopy(self.mlp_head) for i in range(6))
            # trunc_normal_(self.keypoint_token, std=.02)
            # if apply_init:
            #    self.apply(self.init_weights)

            self.loss = build_loss(loss_keypoint)
            self.train_cfg = {} if train_cfg is None else train_cfg
            self.test_cfg = {} if test_cfg is None else test_cfg
            self.target_type = self.test_cfg.get('target_type', 'GaussianHeatMap')
            # self._make_position_embedding(8, 8, dim)
            self._init_layers()

        def _init_layers(self):
            """Initialize layers of the transformer head."""
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

            # self.query_proj = Linear(
            #    self.in_channels, self.embed_dims)

        def init_weights(self):
            """Initialize weights of the transformer head."""
            # The initialization for transformer is important
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
            nn.init.uniform_(self.position.weight.data, 0, 1)

        def _make_sine_position_embedding(self, d_model, temperature=10000,
                                          scale=2 * math.pi):
            h, w = 8, 8
            area = torch.ones(1, h, w)  # [b, h, w]
            y_embed = area.cumsum(1, dtype=torch.float32)
            x_embed = area.cumsum(2, dtype=torch.float32)

            one_direction_feats = d_model // 2

            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

            dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
            dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            pos = pos.flatten(2).permute(0, 2, 1)
            return pos

        def kou(self,x,t):
            x[x < 0.6] = 0
            for i in range(0):
                x[x < 0.25] = 0
                xmin=x.min(dim=-1).values
                xmax= x.max(dim=-1).values
                x=(x-xmin.unsqueeze(-1))/(xmax-xmin).unsqueeze(-1)

            return x
        def forward(self, x, feature_s, target_s, mask_s):

            x = self.input_proj(x)  # bs*2048*8*8


            for i in range(len(feature_s)):
                feature_s[i] = self.input_proj(feature_s[i])
            masks = x.new_zeros((x.shape[0], x.shape[2] * x.shape[3])).to(torch.bool)  # bs*64

            query_embed_list = []
            for feature, target in zip(feature_s, target_s):
                resized_feature = resize(
                    input=feature,
                    size=target.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
                target = target / (target.sum(dim=-1).sum(dim=-1)[:, :, None, None] + 1e-8)
                query_embed = target.flatten(2) @ resized_feature.flatten(2).permute(0, 2, 1)
                query_embed_list.append(query_embed)
            query_embed = torch.mean(torch.stack(query_embed_list, dim=0), 0)  # 2*100*2048
            query_embed = query_embed * mask_s

            masks_ = x.new_zeros((query_embed.shape[0], 1, 100)).to(torch.bool)
            support_order_embedding = self.positional_encoding(
                masks_)

            query_embed = query_embed + support_order_embedding.permute(0, 3, 1, 2).squeeze(-1)
            bs, _, dim = query_embed.shape
            masks_query = (~mask_s.to(torch.bool)).squeeze(-1)  # 2*100

            query_embed=self.GKP(query_embed,torch.cat((query_embed,feature_s[0].view(bs, dim, -1).permute(0, 2, 1),x.view(bs, dim, -1).permute(0, 2, 1)),dim=1),torch.cat((masks_query, masks,masks), dim=1))
            x = x.view(bs, dim, -1).permute(0, 2, 1)  # [bs, c, h, w] -> [bs, h*w, dim]
            b, n, _ = x.shape
            x += self.pos_embedding[:, :n]

            xx = torch.cat((query_embed, x), dim=1)

            mm = torch.cat((masks_query, masks), dim=1)
            xx = self.dropout(xx)
            x = self.transformer(xx, mm,  self.pos_embedding)

            x = self.to_keypoint_token(x[:, 0:self.num_keypoints])
            x = self.mlp_head(x)

            return x

            # return heatmaps

        def cv_squared(self, x):
            """The squared coefficient of variation of a sample.
            Useful as a loss to encourage a positive distribution to be more uniform.
            Epsilons added for numerical stability.
            Epsilons added for numerical stability.
            Returns 0 for an empty Tensor.
            Args:
            x: a `Tensor`.
            Returns:
            a `Scalar`.
            """
            eps = 1e-10
            # if only num_experts = 1

            if x.shape[0] == 1:
                return torch.tensor([0], device=x.device, dtype=x.dtype)
            return   (x.float().var()+eps) /(x.float().mean() ** 2 + eps)

        def get_loss(self, output, target, target_weight, target_sizes):
            """Calculate top-down keypoint loss.
            Args:
                output (torch.Tensor[num_dec_layer x N x K x 2]): Predicted keypoints from each layer of the transformer decoder.
                inital_proposals: Predicted proposals via similarity matching,
                target (torch.Tensor[NxKx2]): Target keypoints.
                target_weight (torch.Tensor[NxKx1]):
                    Weights across different joint types.
                target_sizes (torch.Tensor[Nx2):
                    The image sizes to denomralize the predicted outputs.
            """

            losses = dict()
            # denormalize the predicted coordinates.
            bs, nq = output.shape[:2]
            target_sizes = target_sizes.to(output.device)  # [bs, 1, 2]
            # output = target_sizes * output
            target = target / target_sizes
            # target = target[None, :, :, :].repeat(num_dec_layer, 1, 1, 1)

            # set the weight for unset query point to be zero
            normalizer = target_weight.squeeze(dim=-1).sum(dim=-1)  # [bs, ]
            normalizer[normalizer == 0] = 1

            # compute l1 loss for each layer
            # loss = 0

            layer_output, layer_target = output, target
            l1_loss = F.l1_loss(layer_output, layer_target, reduction="none")  # [bs, query, 2]


            l1_loss = l1_loss.sum(dim=-1, keepdim=False) * target_weight.squeeze(dim=-1)  # [bs, query]
            # normalize the loss for each sample with the number of visible joints
            l1_loss = l1_loss.sum(dim=-1, keepdim=False) / normalizer  # [bs, ]
            losses['l1_loss' + '_layer' + str(0)] = l1_loss.sum() / bs





            # loss += l1_loss.sum() / bs
            # losses['l1_loss'] = loss / num_dec_layer
            return losses

        def get_accuracy(self, output, target, target_weight, target_sizes):
            """Calculate accuracy for top-down keypoint loss.

            Args:
                output (torch.Tensor[NxKx2]): estimated keypoints in ABSOLUTE coordinates.
                target (torch.Tensor[NxKx2]): gt keypoints in ABSOLUTE coordinates.
                target_weight (torch.Tensor[NxKx1]): Weights across different joint types.
                target_sizes (torch.Tensor[Nx2): shapes of the image.
            """
            # NOTE: In POMNet, PCK is estimated on 1/8 resolution, which is slightly different here.

            accuracy = dict()
            output = output * 256.0  # a temporary HARDCODE as all training samples are resized to 256x256
            output, target, target_weight, target_sizes = (
                output.detach().cpu().numpy(), target.detach().cpu().numpy(),
                target_weight.squeeze(-1).long().detach().cpu().numpy(),
                target_sizes.squeeze(1).detach().cpu().numpy())

            _, avg_acc, _ = keypoint_pck_accuracy(
                output,
                target,
                target_weight.astype(np.bool8),
                thr=0.2,
                normalize=target_sizes)
            accuracy['acc_pose'] = float(avg_acc)

            return accuracy

        def decode(self, img_metas, output, img_size, **kwargs):
            """Decode the predicted keypoints from prediction.

            Args:
                img_metas (list(dict)): Information about data augmentation
                    By default this includes:
                    - "image_file: path to the image file
                    - "center": center of the bbox
                    - "scale": scale of the bbox
                    - "rotation": rotation of the bbox
                    - "bbox_score": score of bbox
                output (np.ndarray[N, K, H, W]): model predicted heatmaps.
            """
            batch_size = len(img_metas)
            W, H = img_size
            output = output * np.array([
                W, H
            ])[None, None, :]  # [bs, query, 2], coordinates with recovered shapes.

            if 'bbox_id' or 'query_bbox_id' in img_metas[0]:
                bbox_ids = []
            else:
                bbox_ids = None

            c = np.zeros((batch_size, 2), dtype=np.float32)
            s = np.zeros((batch_size, 2), dtype=np.float32)
            image_paths = []
            score = np.ones(batch_size)
            for i in range(batch_size):
                c[i, :] = img_metas[i]['query_center']
                s[i, :] = img_metas[i]['query_scale']
                image_paths.append(img_metas[i]['query_image_file'])

                if 'query_bbox_score' in img_metas[i]:
                    score[i] = np.array(
                        img_metas[i]['query_bbox_score']).reshape(-1)
                if 'bbox_id' in img_metas[i]:
                    bbox_ids.append(img_metas[i]['bbox_id'])
                elif 'query_bbox_id' in img_metas[i]:
                    bbox_ids.append(img_metas[i]['query_bbox_id'])

            preds = np.zeros(output.shape)
            for idx in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i],
                    c[i],
                    s[i], [W, H],
                    use_udp=self.test_cfg.get('use_udp', False))

            all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
            all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
            all_preds[:, :, 0:2] = preds[:, :, 0:2]
            all_preds[:, :, 2:
                            3] = 1.0  # NOTE: Currently, assume all predicted points are of 100% confidence.
            all_boxes[:, 0:2] = c[:, 0:2]
            all_boxes[:, 2:4] = s[:, 0:2]
            all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
            all_boxes[:, 5] = score

            result = {}

            result['preds'] = all_preds
            result['boxes'] = all_boxes
            result['image_paths'] = image_paths
            result['bbox_ids'] = bbox_ids

            return result

