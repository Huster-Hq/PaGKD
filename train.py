import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.loader import PolyDataset_nocrop_5fold_pair,MultiModalImbalancedDataset,EpochAwareDataLoader,collate_fn
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import argparse
import logging
from torch import Tensor
from itertools import zip_longest
import sys
from torch.utils.tensorboard import SummaryWriter
from models.model_stu import Resnet_att
from models.resnet50 import resnet50
from models.embedded_layer import embed_layer
from utils.PAR import PAR
from utils.camutils import label_to_aff_mask_cross, label_to_aff_mask_cross_set_3,refine_cams_with_bkg_v3
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score


sim_loss_Den=torch.nn.CosineEmbeddingLoss()
sim_loss=torch.nn.CosineEmbeddingLoss(reduction='none')
def try_get_pretrained(teacher, student,  scratch):
    raw_stu_path =''
    tea_path = ''   
    if not scratch:
        if os.path.exists(raw_stu_path):
            student.student.load_state_dict(torch.load(raw_stu_path, map_location=opt.device), strict=False)
            print('load raw_stu_path compeleted')
        if os.path.exists(tea_path):
            teacher.load_state_dict(torch.load(tea_path, map_location=opt.device), strict=False)
            print('load teacher compeleted')
    return teacher, student

def get_ce_loss(img, label, network):
    pred,x_embed, x4= network.student_out(img)
    errCE = ce_loss_func(pred, label)
    pred_label = torch.argmax(pred, dim=-1)
    acc = (pred_label == label).sum().float() / pred_label.shape[0]
    acc_all = (pred_label == label).sum().float()
    return errCE, acc, pred, x4, x_embed, acc_all


def save_model(epoch):
    print('update model..')
    stu_path = opt.train_save+'/weights/poly_student_model-{}.pth'.format(epoch)
    torch.save(student.state_dict(), stu_path)

def cam(fmaps,model_predict,cls_idx):
    score =model_predict[:,0]
    for pp in range(cls_idx.shape[0]):
        score[pp]=model_predict[pp, cls_idx[pp]]
    score = score.sum()
    weights = torch.autograd.grad(outputs=score, inputs=fmaps,create_graph=True)[0] 
    weights = weights.mean(dim=(2, 3))
    grad_cam = (weights.view(*weights.shape, 1, 1) * fmaps.squeeze(0)).sum(1)

    def _normalize(cams: Tensor) -> Tensor:
        """CAM normalization"""
        cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
        return cams
    B, H, W = grad_cam.shape
    grad_cam = _normalize(F.relu(grad_cam, inplace=True))
    # # scale gradcam to image size
    grad_cam = grad_cam.view(B,1,H,W)
    return grad_cam, weights

def Den(pseudo_label_aux1, pseudo_label_aux2,q,kv):
    aff_mask = label_to_aff_mask_cross(pseudo_label_aux1, pseudo_label_aux2, ignore_index=opt.ignore_index, confuse_value=0.5)
    k = kv
    v = kv.transpose(2,1)
    matmul_qk = torch.matmul(q.transpose(2,1), k)
    matmul_qk_= matmul_qk*aff_mask
    matmul_qk_= matmul_qk_.masked_fill(aff_mask == 0, -1e9)
    attmap = F.softmax(matmul_qk_, dim=-1)
    f_cam_att = torch.matmul(attmap, v)               
    loss_align=sim_loss_Den(f_cam_att.transpose(1,2).flatten(1),q.flatten(1), torch.ones(1).to(opt.device))
    return loss_align

def Den1(aff_mask,q,kv):
    k = kv
    v = kv
    q=q.unsqueeze(1).repeat([1,q.shape[0],1,  1])
    matmul_qk= torch.einsum('mnct,ncv->mntv', [q, k])
    matmul_qk_= matmul_qk*aff_mask
    matmul_qk_= matmul_qk_.masked_fill(aff_mask == 0, -1e9)
    attmap = F.softmax(matmul_qk_, dim=-1)
    f_cam_att = torch.einsum('mntv,ncv->mnct', [attmap, v])
    return f_cam_att     

def Den_set(pseudo_label_aux1, pseudo_label_aux2,q,kv,idx0,idx1,idx2):
    kv_0 = torch.index_select(kv, 0, idx0[:,0])
    q_0 = torch.index_select(q, 0, idx0[:,0])
    kv_1 = torch.index_select(kv, 0, idx1[:,0])
    q_1 = torch.index_select(q, 0, idx1[:,0])
    kv_2 = torch.index_select(kv, 0, idx2[:,0])
    q_2 = torch.index_select(q, 0, idx2[:,0])

    aff_mask0,aff_mask1,aff_mask2 = label_to_aff_mask_cross_set_3(idx0,idx1,idx2,pseudo_label_aux1, pseudo_label_aux2, ignore_index=opt.ignore_index, confuse_value=0.5)#resize后的cam相当于每个patch的代表点,乘的顺序需要和crossattention对应

    b0 = q_0.shape[0]
    b1= q_1.shape[0]
    b2= q_2.shape[0]

    f_cam_att0=Den1(aff_mask0,q_0,kv_0)
    f_cam_att1=Den1(aff_mask1,q_1,kv_1)
    f_cam_att2=Den1(aff_mask2,q_2,kv_2)              
    loss_align0=sim_loss(f_cam_att0.reshape(b0 * b0, -1),q_0.expand_as(f_cam_att0).reshape(b0 * b0, -1), torch.ones(b0 * b0).to(opt.device))
    loss_align1=sim_loss(f_cam_att1.reshape(b1 * b1, -1),q_1.expand_as(f_cam_att1).reshape(b1 * b1, -1), torch.ones(b1 * b1).to(opt.device))
    loss_align2=sim_loss(f_cam_att2.reshape(b2 * b2, -1),q_2.expand_as(f_cam_att2).reshape(b2 * b2, -1), torch.ones(b2 * b2).to(opt.device))
    loss_align=loss_align0.mean()+loss_align1.mean()+loss_align2.mean()

    return loss_align



def softmax_1(x):
    """
    (softmax_1(x))_i = exp(x_i) / (1 + sum_j exp(x_j))
    """
    exp_x = torch.exp(x)
    sum_exp = torch.sum(exp_x)
    return exp_x / (1 + sum_exp)


def set_loss(token_embeddings, labels, tmp=1):
    labels = torch.tensor(labels).to(token_embeddings.device)
    loss = 0.0
    feat = token_embeddings/ token_embeddings.norm(dim=-1, keepdim=True)

    matrix0 = torch.einsum('ct,bcv->btv', [feat[0], feat[1:]])
    matrix0 = torch.diagonal(matrix0, dim1=1, dim2=2)
    S0 = torch.sum(matrix0, dim=-1)/20

    matrix1 = torch.einsum('ct,bcv->btv', [feat[2], feat[[0,1,3,4,5]]])
    matrix1 = torch.diagonal(matrix1, dim1=1, dim2=2)
    S1 = torch.sum(matrix1, dim=-1)/20

    matrix2 = torch.einsum('ct,bcv->btv', [feat[4], feat[[0,1,2,3,5]]])
    matrix2 = torch.diagonal(matrix2, dim1=1, dim2=2)
    S2 = torch.sum(matrix2, dim=-1)/20

    logpt0=softmax_1(S0/tmp)
    logpt1=softmax_1(S1/tmp)
    logpt2=softmax_1(S2/tmp)


    logpt0 = torch.log(logpt0)
    logpt1 = torch.log(logpt1)
    logpt2 = torch.log(logpt2)
    loss =logpt0[0]+logpt1[2]+logpt2[-1]

    return -loss


def calculate_metrics(y_true, y_pred):
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        return metrics

def pair_loss(pred_wli,x4_stu, f_stu_att, pred_prob_pos,x4_tea,f_tea_att, par,label_pair,white_img_pair,nbi_img_pair):
    cam1, _ = cam(x4_stu, pred_wli, label_pair, opt.device)#
    cam2, _ = cam(x4_tea, pred_prob_pos, label_pair, opt.device)

    cls_labels = torch.ones(size=(cam1.shape[0],1)).to(opt.device)
    pseudo_label_aux1 = refine_cams_with_bkg_v3(par, white_img_pair, cams=cam1, cls_labels=cls_labels, cfg=opt)
    pseudo_label_aux2 = refine_cams_with_bkg_v3(par, nbi_img_pair, cams=cam2, cls_labels=cls_labels, cfg=opt)
    loss_Den_1=Den(pseudo_label_aux1, pseudo_label_aux2,f_tea_att,f_stu_att)
    loss_Den_2=Den(pseudo_label_aux2,pseudo_label_aux1,f_stu_att,f_tea_att)
    return loss_Den_1+loss_Den_2

def unpair_loss(pred_wli,x4_stu, f_stu_att, pred_prob_pos,x4_tea,f_tea_att, par,label_unpair,white_img_unpair,nbi_img_unpair,idx0,idx1,idx2):
    set_fea, set_tokens, set_pred_logit = student(f_stu_att, f_tea_att, label_unpair)
    set_cls_loss = ce_loss_func(set_pred_logit, torch.tensor([0, 0, 1, 1,2,2]).to(opt.device).long())
    set_contrastive_loss = set_loss(set_fea, [0,0,1,1,2,2])
    cam_w, _ = cam(x4_stu, pred_wli, label_unpair, opt.device)
    cam_n, _ = cam(x4_tea, pred_prob_pos, label_unpair, opt.device)
    cls_labels = torch.ones(size=(cam_w.shape[0],1)).to(opt.device)
    pseudo_label_aux_w = refine_cams_with_bkg_v3(par, white_img_unpair, cams=cam_w, cls_labels=cls_labels, cfg=opt)
    pseudo_label_aux_n = refine_cams_with_bkg_v3(par, nbi_img_unpair, cams=cam_n, cls_labels=cls_labels, cfg=opt)
    loss_Den_unpair_1=Den_set(pseudo_label_aux_w, pseudo_label_aux_n,f_tea_att,f_stu_att,idx0,idx1,idx2)
    loss_Den_unpair_2=Den_set(pseudo_label_aux_n,pseudo_label_aux_w,f_stu_att,f_tea_att,idx0,idx1,idx2)
    space_loss_unpair=loss_Den_unpair_1+loss_Den_unpair_2
    return set_cls_loss,set_contrastive_loss,space_loss_unpair

def train(teacher, student,embed_layer_, epochs=1000, is_test=True):
    optimizer_stu = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-8)
    optimizer_embed_layer=torch.optim.Adam(embed_layer_.parameters(), lr=1e-4, weight_decay=1e-8)
    par = PAR(num_iter=10, dilations=[1,2,4,8])
    par.to(opt.device)
    prev_best = 0

    if is_test:
        phases = ('test',)
    else:
        phases = ('train', 'test')#

    for epoch in range(1, epochs):
        for phase in iter(phases):
            if phase == 'train':
                teacher.eval()
                student.train()
                unpair_ldr = unpair_loader
                pair_ldr=pair_loader

            else:
                teacher.eval()
                student.eval()
                pair_ldr = val_loader

            all_preds=[]
            all_labels=[]
            
            if phase == 'train':
                total_len = max(len(pair_ldr), len(unpair_ldr))
                for batch_pair, batch_unpair in tqdm(zip_longest(pair_ldr, unpair_ldr),total=total_len, desc="Processing"):

                    space_loss_unpair=torch.zeros(1).to(opt.device)
                    space_loss_pair=torch.zeros(1).to(opt.device)
                    set_contrastive_loss=torch.zeros(1).to(opt.device)
                    set_cls_loss=torch.zeros(1).to(opt.device)
                    errCE_wht_pair=torch.zeros(1).to(opt.device)
                    errCE_wht_unpair=torch.zeros(1).to(opt.device)
                    if batch_pair is not None:
                        white_img_pair, nbi_img_pair, label_pair,filename= \
                            batch_pair[0].to(opt.device).float(), batch_pair[1].to(opt.device).float(), batch_pair[2].to(opt.device).long(), batch_pair[3][0]   

                        pred_prob_pos_pair,_,x4_tea_pair= teacher(nbi_img_pair)
                        errCE_wht_pair, _, pred_wli_pair, x4_stu_pair,f_stu_att_pair,_ = get_ce_loss(white_img_pair, label_pair, student)
                        pred_label_pair = torch.argmax(pred_wli_pair, dim=-1)
                        all_preds.extend(pred_label_pair.cpu().numpy())
                        all_labels.extend(label_pair.cpu().numpy())
                        f_tea_att_pair=embed_layer_(x4_tea_pair.detach())
                        optimizer_stu.zero_grad()
                        optimizer_embed_layer.zero_grad()
                        if epoch>5:
                            space_loss_pair=pair_loss(pred_wli_pair, x4_stu_pair,f_stu_att_pair,pred_prob_pos_pair,x4_tea_pair,f_tea_att_pair,par,label_pair,white_img_pair,nbi_img_pair)
                        loss_pair=errCE_wht_pair+space_loss_pair

                        if not torch.isfinite(loss_pair):
                            print('WARNING: non-finite loss, ending training ', loss_pair)
                            sys.exit(1)
                        loss_pair.backward()
                        optimizer_embed_layer.step()
                        optimizer_stu.step()
                    if batch_unpair is not None:
                        white_img_unpair, nbi_img_unpair, label_unpair=\
                            batch_unpair['WL'].to(opt.device).float(),batch_unpair['NBI'].to(opt.device).float(),batch_unpair['class'].to(opt.device).long()   
                        optimizer_stu.zero_grad()
                        optimizer_embed_layer.zero_grad()
                        pred_prob_pos_unpair,_,x4_tea_unpair= teacher(nbi_img_unpair)
                        errCE_wht_unpair, _, pred_wli_unpair, x4_stu_unpair,f_stu_att_unpair,_ = get_ce_loss(white_img_unpair, label_unpair, student)#f_stu_umap,cam_wl
                        pred_label_unpair = torch.argmax(pred_wli_unpair, dim=-1)
                        all_preds.extend(pred_label_unpair.cpu().numpy())
                        all_labels.extend(label_unpair.cpu().numpy())

                        f_tea_att_unpair=embed_layer_(x4_tea_unpair.detach())
                        # # #set_loss
                        idx0=torch.nonzero(label_unpair == 0)
                        idx1=torch.nonzero(label_unpair == 1)
                        idx2=torch.nonzero(label_unpair == 2)
                        set_cls_loss=torch.zeros(1).to(opt.device)
                        
                        if len(idx0[:,0])>0 and len(idx1[:,0])>0 and len(idx2[:,0])>0 and epoch>5:
                            set_cls_loss,set_contrastive_loss,space_loss_unpair=unpair_loss(pred_wli_unpair,x4_stu_unpair, f_stu_att_unpair, pred_prob_pos_unpair,x4_tea_unpair,f_tea_att_unpair, par,label_unpair,white_img_unpair,nbi_img_unpair,idx0,idx1,idx2)
                        loss_unpair=errCE_wht_unpair+space_loss_unpair+0.001*(set_contrastive_loss+set_cls_loss)
                        if not torch.isfinite(loss_unpair):
                            print('WARNING: non-finite loss, ending training ', loss_unpair)
                            sys.exit(1)
                        loss_unpair.backward()
                        errCE_wht=errCE_wht_unpair+errCE_wht_pair

                        optimizer_embed_layer.step()
                        optimizer_stu.step()


            else:
                for batch in tqdm(pair_ldr, desc="Processing"):
                    white_img, nbi_img, label,filename= \
                        batch[0].to(opt.device).float(), batch[1].to(opt.device).float(), batch[2].to(opt.device).long(), batch[3][0] 
                    errCE_wht_val, _, pred_wli,_,_,_ = get_ce_loss(white_img, label, student)
                    pred_label = torch.argmax(pred_wli, dim=-1)
                    all_preds.extend(pred_label.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())
                
            metrics = calculate_metrics(
                np.array(all_labels), 
                np.array(all_preds),
            )

            if phase == 'train':
                print('Epoch %d' % epoch,' CE_loss: %0.2f,  tran_acc: %0.2f,w_pre:%0.2f,w_recall:%0.2f,w_F1:%0.2f' %
                       (errCE_wht.item(), metrics['accuracy'],metrics['precision_weighted'],metrics['recall_weighted'],metrics['f1_weighted'] ))#
                save_model(epoch)

            else:
                logging.info('epoch: {},  val_acc:{},w_pre:{},w_recall:{},w_F1:{}'.format(epoch, metrics['accuracy'],metrics['precision_weighted'],metrics['recall_weighted'],metrics['f1_weighted']))
                print('[EVAL] Epoch %d' % epoch, 'CE_loss: %0.2f, val_acc: %0.2f,w_pre:%0.2f,w_recall:%0.2f,w_F1:%0.2f, best_f1: %0.3f' %
                       (errCE_wht_val.item(), metrics['accuracy'],metrics['precision_weighted'],metrics['recall_weighted'],metrics['f1_weighted'], prev_best))                
                if metrics['f1_weighted'] > prev_best :
                    prev_best = metrics['f1_weighted']
                    print('##############################################################################best', prev_best)
                    logging.info('##############################################################################best_val:{}'.format(prev_best)) 
        


if __name__ == '__main__':
    for i in range(5):
        is_test = False
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_save', type=str, default='/memory/wangqimei/PolypsAlign-main/log_ADDset_piccolo/test/'+str(i))#自定义保存路径
        parser.add_argument('--fold', type=int,
                            default=i)
        parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument("--bs", default=24, type=int, help="batch_size")        
        parser.add_argument("--n_cls", default=3, type=int, help="classes number")        

        parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
        parser.add_argument("--low_thre", default=0.3, type=float, help="low_bkg_score")
        parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
        parser.add_argument("--ignore_index", default=255, type=int, help="random index")
        opt = parser.parse_args()
        if os.path.exists(opt.train_save+'/run/') is False:
            os.makedirs(opt.train_save+'/run/')
        if os.path.exists(opt.train_save+'/weights/') is False:
            os.makedirs(opt.train_save+'/weights/')    
        logging.basicConfig(filename=opt.train_save+'/train_log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

        unpair_dataset = MultiModalImbalancedDataset(split_id=opt.fold)
        
        sampler = unpair_dataset.get_balanced_sampler()
        

        unpair_loader = EpochAwareDataLoader(
        unpair_dataset,
        batch_size=opt.bs,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
        )
        pair_dataset = PolyDataset_nocrop_5fold_pair(is_train=True,split_id=opt.fold)
        val_dataset = PolyDataset_nocrop_5fold_pair(is_train=False,split_id=opt.fold)
        pair_loader = DataLoader(pair_dataset, batch_size=opt.bs, num_workers=opt.bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, num_workers=opt.bs, shuffle=False)
        ce_loss_func = nn.CrossEntropyLoss()
        student = Resnet_att(num_classes=opt.n_cls).to(opt.device)
        teacher = resnet50(pretrained=True, num_classes=opt.n_cls).to(opt.device)
        embed_layer_=embed_layer().to(opt.device)
        teacher, student = try_get_pretrained(teacher, student,  scratch=False)
        tb_writer = SummaryWriter(opt.train_save+'/run/')
        train(teacher, student,embed_layer_, is_test=is_test,epochs=200)

