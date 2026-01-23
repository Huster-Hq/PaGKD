
import torch
import torch.nn.functional as F

def label_to_aff_mask_cross(cam_label1,cam_label2, ignore_index=255,confuse_value=0.5):
    b = cam_label1.shape[0]
    _cam_label1 = cam_label1.reshape(b, 1, -1)
    _cam_label_rep1 = _cam_label1.repeat([1, _cam_label1.shape[-1], 1])
    _cam_label2 = cam_label2.reshape(b, 1, -1)
    _cam_label_rep2 = _cam_label2.repeat([1, _cam_label2.shape[-1], 1])
    _cam_label_rep2_t = _cam_label_rep2.permute(0,2,1)
    aff_label = (_cam_label_rep1 == _cam_label_rep2_t).type(torch.float)
    
    for i in range(b):
        aff_label[i, :, _cam_label_rep1[i, 0, :]==ignore_index] = confuse_value
        aff_label[i, _cam_label_rep2[i, 0, :]==ignore_index, :] = confuse_value
    return aff_label

def label_to_aff_mask_cross1(cam_label1,cam_label2, ignore_index=255,confuse_value=0.5):
    b,_,h,w = cam_label1.shape

    _cam_label1 = cam_label1.reshape(b, 1,1, -1)
    _cam_label_rep1 = _cam_label1.repeat([1,1, _cam_label1.shape[-1], 1])
    _cam_label2 = cam_label2.reshape(b, 1, -1)
    _cam_label_rep2 = _cam_label2.repeat([1, _cam_label2.shape[-1], 1])
    _cam_label_rep2_t = _cam_label_rep2.permute(0,2,1)
    aff_label = (_cam_label_rep1 == _cam_label_rep2_t).type(torch.float)
    for i in range(b):
        aff_label[i,:, :, _cam_label_rep1[i,0, 0, :]==ignore_index] = confuse_value
        aff_label[:,i, _cam_label_rep2[i, 0, :]==ignore_index, :] = confuse_value
    return aff_label

def label_to_aff_mask_cross_set(idx0,idx1,cam_label1,cam_label2, ignore_index=255,confuse_value=0.5):
    cam_label1_0 = torch.index_select(cam_label1, 0, idx0[:,0])
    cam_label2_0 = torch.index_select(cam_label2, 0, idx0[:,0])
    cam_label1_1 = torch.index_select(cam_label1, 0, idx1[:,0])
    cam_label2_1 = torch.index_select(cam_label2, 0, idx1[:,0])
    aff_label0=label_to_aff_mask_cross1(cam_label1_0.unsqueeze(dim=1),cam_label2_0, ignore_index=ignore_index,confuse_value=confuse_value)
    aff_labe1=label_to_aff_mask_cross1(cam_label1_1.unsqueeze(dim=1),cam_label2_1, ignore_index=ignore_index,confuse_value=confuse_value)

    return aff_label0,aff_labe1

def label_to_aff_mask_cross_set_3(idx0,idx1,idx2,cam_label1,cam_label2, ignore_index=255,confuse_value=0.5):
    cam_label1_0 = torch.index_select(cam_label1, 0, idx0[:,0])
    cam_label2_0 = torch.index_select(cam_label2, 0, idx0[:,0])
    cam_label1_1 = torch.index_select(cam_label1, 0, idx1[:,0])
    cam_label2_1 = torch.index_select(cam_label2, 0, idx1[:,0])
    cam_label1_2 = torch.index_select(cam_label1, 0, idx2[:,0])
    cam_label2_2 = torch.index_select(cam_label2, 0, idx2[:,0])
    aff_label0=label_to_aff_mask_cross1(cam_label1_0.unsqueeze(dim=1),cam_label2_0, ignore_index=ignore_index,confuse_value=confuse_value)
    aff_labe1=label_to_aff_mask_cross1(cam_label1_1.unsqueeze(dim=1),cam_label2_1, ignore_index=ignore_index,confuse_value=confuse_value)
    aff_labe2=label_to_aff_mask_cross1(cam_label1_2.unsqueeze(dim=1),cam_label2_2, ignore_index=ignore_index,confuse_value=confuse_value)
    return aff_label0,aff_labe1,aff_labe2    

def _refine_cams(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label

def refine_cams_with_bkg_v3(ref_mod=None, images=None, cams=None, cls_labels=None, cfg=None,  down_scale=2):#img_box=None,
    _,_,h1,w1 = images.shape
    b,_,h,w=cams.shape
    h1=h1//down_scale
    w1=w1//down_scale
    _images = F.interpolate(images, size=[h1,w1], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b,1,h,w))*cfg.high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b,1,h,w))*cfg.low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b,1))
    
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * cfg.ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()
    
    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h,w], mode="bilinear", align_corners=False)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h,w], mode="bilinear", align_corners=False)

    for idx, _ in enumerate(images):
        valid_key = torch.nonzero(cls_labels[idx,...])[:,0]
        valid_cams_h = _cams_with_bkg_h[idx, ...].unsqueeze(0).softmax(dim=1)#
        valid_cams_l = _cams_with_bkg_l[idx, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_h, valid_key=valid_key, orig_size=(h,w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_l, valid_key=valid_key, orig_size=(h,w))
        
        refined_label_h[idx, :, :] = _refined_label_h[0, :, :]
        refined_label_l[idx, :, :] = _refined_label_l[0, :, :]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = cfg.ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label
