import torch
import torch.nn as nn
from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import new_class_names_coco, BACKGROUND_CATEGORY_COCO
from pytorch_grad_cam import GradCAM
from clip.clip_tool import perform_single_coco_cam, generate_cam_label, generate_clip_fts
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP_model.PAR import PAR


def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()

def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


class WeCLIP(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, embedding_dim=256, in_channels=512, dataset_root_path=None, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.encoder, _ = clip.load(clip_model, device=device)
        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes)
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)

        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_COCO, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model)
        self.fg_text_features = zeroshot_classifier(new_class_names_coco, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model) (20, 512)


        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform)

        self.root_path = os.path.join(dataset_root_path, 'SegmentationClass')

        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
        self.iter_num = 0
        self.require_all_fts = True


    # def get_param_groups(self):

    #     param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

    #     for param in list(self.decoder.parameters()):
    #         param_groups[3].append(param)
    #     for param in list(self.decoder_fts_fuse.parameters()):
    #         param_groups[3].append(param)

    #     return param_groups
    
    def get_param_groups(self):
        # backbone; backbone_norm; cls_head; seg_head; comer_modules;
        param_groups = [[], [], [], [], []]  

        # 기존 decoder 파라미터
        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        
        # ViT-Comer 관련 파라미터 추가 (param_groups[4]에 저장)
        
        # CNN backbone (SPM) 파라미터
        for param in list(self.encoder.visual.spm.parameters()):
            param_groups[4].append(param)
        
        # MRFP 모듈 파라미터
        for module in self.encoder.visual.mrfp_modules:
            for param in module.parameters():
                param_groups[4].append(param)
        
        # CTI_toV 모듈 파라미터
        for module in self.encoder.visual.cti_to_v_modules:
            for param in module.parameters():
                param_groups[4].append(param)
        
        # CTI_toC 모듈 파라미터
        for module in self.encoder.visual.cti_to_c_modules:
            for param in module.parameters():
                param_groups[4].append(param)
        
        # Adapter 모듈 파라미터
        for adapter in self.encoder.visual.adapters_to_v:
            for param in adapter.parameters():
                param_groups[4].append(param)
        
        for adapter in self.encoder.visual.adapters_to_c:
            for param in adapter.parameters():
                param_groups[4].append(param)
        
        # Final Conv 파라미터
        for param in self.encoder.visual.final_conv.parameters():
            param_groups[4].append(param)
        
        # Normalization 레이어 파라미터
        for param in self.encoder.visual.norm1.parameters():
            param_groups[4].append(param)
        for param in self.encoder.visual.norm2.parameters():
            param_groups[4].append(param)
        for param in self.encoder.visual.norm3.parameters():
            param_groups[4].append(param)
        for param in self.encoder.visual.norm4.parameters():
            param_groups[4].append(param)
        
        # Upsampling 레이어 파라미터
        for param in self.encoder.visual.up.parameters():
            param_groups[4].append(param)

        return param_groups

    def forward(self, img, img_names, mode='train'):
        # 이미지를 half precision으로 변환
        img = img.half()
        
        cam_list = []
        b, c, h, w = img.shape
        self.iter_num += 1

        # 기존 코드를 수정하여 CTI 출력을 직접 사용
        last_vit_output, transformer_features, cti_outputs, multi_level_features, final_cti, attn_weight_list = self.encoder.visual(
            img, h, w, require_all_fts=True)
        
        # CTI 출력은 이미 8개로 제한되어 있음 (VisionTransformer에서 처리)
        
        # attention weight 처리
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        
        # CAM 관련 처리
        if self.require_all_fts == True:
            cam_fts_all = transformer_features[-1].unsqueeze(0).permute(2, 1, 0, 3)
        else:
            fts_all_stack = torch.stack(transformer_features, dim=0)
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)
        
        # decoder_fts_fuse 대신 decoder 직접 사용
        # final_cti는 이미 적절한 형태로 변환되어 있어야 함 (확인 필요)
        seg, seg_attn_weight_list = self.decoder(final_cti)
        
        # 여기서 중요한 변경: decoder_fts_fuse 사용하지 않고 직접 decoder에 CTI 출력 전달
        # 기존:
        # all_img_tokens = fts_all_stack[:, 1:, ...]
        # all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h // 16, w // 16)
        # fts = self.decoder_fts_fuse(all_img_tokens)
        # seg, seg_attn_weight_list = self.decoder(fts)

        fts = self.decoder_fts_fuse(all_img_tokens)
        attn_fts = fts.clone()
        _, _, fts_h, fts_w = fts.shape

        seg, seg_attn_weight_list = self.decoder(fts)

        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        if mode=='val':
            return seg, None, attn_pred

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.root_path, 'train', str(img_name)+'.png')
            img_i = img[i]
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]
            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]

            if self.iter_num > 40000 or mode=='val': #40000
                require_seg_trans = True
            else:
                require_seg_trans = False

            cam_refined_list, keys, w, h = perform_single_coco_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                   self.bg_text_features, self.fg_text_features,
                                                                   self.grad_cam,
                                                                   mode=mode,
                                                                   require_seg_trans=require_seg_trans)

            cam_dict = generate_cam_label(cam_refined_list, keys, w, h)

            cams = cam_dict['refined_cam'].cuda()

            bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()
            cams = torch.cat([bg_score, cams], dim=0).cuda()

            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).cuda()

            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)

            cam_list.append(cam_labels)

        all_cam_labels = torch.stack(cam_list, dim=0)


        return seg, all_cam_labels, attn_pred


