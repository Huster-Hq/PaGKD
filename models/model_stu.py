from models.resnet50 import resnet50
import torch
import torch.nn as nn
import math

class classifier(nn.Module):
    def __init__(self,input_dim, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x_ = self.avgpool(x)
        x = torch.flatten(x_, 1)
        x = self.fc(x)
        return x, x_
    
class Resnet_att(nn.Module):

    def __init__(self, embed_dim=128,num_classes=2):
        super(Resnet_att, self).__init__()
        self.student = resnet50(pretrained=True, num_classes=num_classes)
        self.lb_token1 = torch.nn.Parameter(torch.randn(12, 128))
        self.CA = nn.MultiheadAttention(embed_dim, num_heads=16, batch_first=True)
        self.cls = classifier(embed_dim, num_classes)
    

    def generate_identity(self, *batch_sizes):
        encodings = []
        num=[42,57]
        for i,size in enumerate(batch_sizes):
            if i ==0:
                encoding = torch.tensor([math.sin((i+0.1) * 0.1) for i in range(size)])
            else:
                torch.manual_seed(num[i-1])
                encoding = torch.rand(size)
            encodings.append(encoding.unsqueeze(1).unsqueeze(2))
        return encodings

    def student_out(self, x):
        pred_wli,f_wli, x4_wli= self.student(x)
        return pred_wli,f_wli, x4_wli
    
    def forward(self, f_wli, f_nbi, label):
        f_nbi = f_nbi.transpose(1,2)
        f_wli = f_wli.transpose(1,2)
        _, d, c = f_nbi.shape
        label_classes = torch.unique(label)
        label_classes, _ = torch.sort(label_classes)
        wli_feaset_dict = {}
        nbi_feaset_dict = {}
        wli_fea_dict = {}
        nbi_fea_dict = {}
        batch_sizes = []

        for cls in label_classes:
            cls = cls.item()
            idx = torch.nonzero(label == cls, as_tuple=False).squeeze(1)
            idx = idx.to(f_wli.device)

            wli_feats = torch.index_select(f_wli, 0, idx) 
            nbi_feats = torch.index_select(f_nbi, 0, idx)

            batch_sizes.append(wli_feats.shape[0])

            wli_feaset_dict[cls] = wli_feats
            nbi_feaset_dict[cls] = nbi_feats


        fixed_encodings = self.generate_identity(*batch_sizes)


        for i, cls in enumerate(label_classes):
            cls = cls.item()

            fixed_enc = fixed_encodings[i].expand(-1, d, c).to(f_wli.device)

            wli = wli_feaset_dict[cls] + fixed_enc
            nbi = nbi_feaset_dict[cls] + fixed_enc

            wli_flat = wli.reshape(-1, c).unsqueeze(0)
            nbi_flat = nbi.reshape(-1, c).unsqueeze(0)

            q = self.lb_token.unsqueeze(0) 

            wli_fea_dict[cls], _ = self.CA(q, wli_flat, wli_flat)
            nbi_fea_dict[cls], _ = self.CA(q, nbi_flat, nbi_flat)

        for cls in label_classes:
            if cls.item()==0:
                all_fea=torch.cat((wli_fea_dict[cls.item()],nbi_fea_dict[cls.item()]), dim=0)
            else:
                all_fea=torch.cat((all_fea,wli_fea_dict[cls.item()]), dim=0)
                all_fea=torch.cat((all_fea,nbi_fea_dict[cls.item()]), dim=0)

        set_pred_logit, set_tokens  = self.cls(all_fea.transpose(1,2)) 
        return all_fea.transpose(1,2),set_tokens, set_pred_logit 
