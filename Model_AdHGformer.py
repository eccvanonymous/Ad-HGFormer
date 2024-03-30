import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GAE
from numpy.lib.shape_base import kron
from sklearn.cluster import KMeans
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges



###################---------Misc---------###################
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. " "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
###################---------Misc---------###################



###############---------Class_Import---------###############
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
###############---------Class_Import---------###############



#############-------Weight_Initialization-------#############
def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
#############-------Weight_Initialization-------#############



##%%%%%%%%%%%%%%%%%%%%--------TCN--------%%%%%%%%%%%%%%%%%%%%##
###############---------T_Convolution---------###############
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1),)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
###############---------T_Convolution---------###############

##################--------TCN_main()--------##################
class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=False,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        #Update_IntBranch&KernelSize
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        
        # Temporal Convolution branches
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0), 
                                                     nn.BatchNorm2d(branch_channels), 
                                                     nn.ReLU(inplace=True),
                                                     TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),)
                                                     for ks, dilation in zip(kernel_size, dilations)])
        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                                           nn.BatchNorm2d(branch_channels),
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
                                           nn.BatchNorm2d(branch_channels)))

        self.branches.append(nn.Sequential(nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
                                           nn.BatchNorm2d(branch_channels)))

        #Residual Connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out
##################--------TCN_main()--------##################
##%%%%%%%%%%%%%%%%%%%%--------TCN--------%%%%%%%%%%%%%%%%%%%%##



##################-----------Residual_TCN-----------##################
class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1), groups=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
##################-----------Residual_TCN-----------##################





###########%%%%%%%%%%%%%%%%%-------------VIT-------------%%%%%%%%%%%%%%%%%###########
##################-----------MLP-----------##################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., num_heads=None):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x
##################-----------MLP-----------##################



##################-----------MHSA-----------##################
class MHSA(nn.Module):

    def __init__(self, dim_in, dim, A, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., 
                 proj_drop=0., insert_cls_layer=0, pe=False, num_point=25, outer=True, layer=0,**kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_point = num_point
        self.layer = layer
        
        h1 = A.sum(0)
        h1[h1 != 0] = 1
        h = [None for _ in range(num_point)]
        h[0] = np.eye(num_point)
        h[1] = h1
        self.hops = 0*h[0]
        for i in range(2, num_point):
            h[i] = h[i-1] @ h1.transpose(0, 1)
            h[i][h[i] != 0] = 1
        for i in range(num_point-1, 0, -1):
            if np.any(h[i]-h[i-1]):
                h[i] = h[i] - h[i - 1]
                self.hops += i*h[i]
            else:
                continue

        self.hops = torch.tensor(self.hops).long()
        self.rpe = nn.Parameter(torch.zeros((self.hops.max()+1, dim)))
        self.w1 = nn.Parameter(torch.zeros(num_heads, head_dim))
        A = A.sum(0)
        A[:, :] = 0
        self.outer = nn.Parameter(torch.stack([torch.eye(A.shape[-1]) for _ in range(num_heads)], dim=0), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.kv = nn.Conv2d(dim_in, dim * 2, 1, bias=qkv_bias)
        self.q = nn.Conv2d(dim_in, dim, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, groups=6)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        self.insert_cls_layer = insert_cls_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, e):
        N, C, T, V = x.shape
        kv = self.kv(x).reshape(N, 2, self.num_heads, self.dim // self.num_heads, T, V).permute(1, 0, 4, 2, 5, 3)
        k, v = kv[0], kv[1]
        ## n t h v c
        q = self.q(x).reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)
        e_k = e.reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)
        pos_emb = self.rpe[self.hops]
        k_r = pos_emb.view(V, V, self.num_heads, self.dim // self.num_heads)
        b = torch.einsum("bthnc, nmhc->bthnm", q, k_r)
        c = torch.einsum("bthnc, bthmc->bthnm", q, e_k)
        d = torch.einsum("hc, bthmc->bthm", self.w1, e_k).unsqueeze(-2)
        a = q @ k.transpose(-2, -1)
        attn = a + b + c + d
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (self.alpha * attn + self.outer) @ v
        # x = attn @ v
        x = x.transpose(3, 4).reshape(N, T, -1, V).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
##################-----------MHSA-----------##################




##########-------Hypergraph_Temporal_Attention-------##########
class hyp_temp_attn(nn.Module):

    def __init__(self, num_feat, squeeze_factor=5):
        super(hyp_temp_attn, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)                                        #[128, 216, 16, 25]->[128, 216, 1, 1]
        self.lin1 = nn.Linear(num_feat, num_feat // squeeze_factor, bias=True)  #[128, 216, 1, 1]->[128, 13, 1, 1]
        self.act = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(num_feat // squeeze_factor, num_feat, bias=True)  #[128, 13, 1, 1]->[128, 216, 1, 1]
        self.sig = nn.Sigmoid()

    def forward(self, x):                  #[128, 25, 216, 16]
        x_mod = self.pool(x)               #[128, 25, 216, 16]->[128, 25, 1, 1]
        x_mod = x_mod.permute(0, 2, 3, 1)  #[128, 25, 1, 1]->[128, 1, 1, 25]
        x_mod = self.lin1(x_mod)           #[128, 1, 1, 25]->[128, 1, 1, 5]
        x_mod = self.act(x_mod)            #[128, 1, 1, 5]->[128, 1, 1, 5]
        x_mod = self.lin2(x_mod)           #[128, 1, 1, 5]->[128, 1, 1, 25]
        x_mod = self.sig(x_mod)            #[128, 1, 1, 25]->[128, 1, 1, 25]

        return x.permute(0, 2, 3, 1) * x_mod    #[128, 216, 16, 25]*[128, 1, 1, 25]->[128, 216, 16, 25]



class hyp_temp_attn_process(nn.Module):

    def __init__(self, num_feat, compress_ratio=4, squeeze_factor=16):
        super(hyp_temp_attn_process, self).__init__()

        # self.conv1 = nn.Conv1d(num_feat, num_feat // compress_ratio, kernel_size=3, stride=1, padding=1)  #[128, 216, 16, 25]->128, 54, 16, 25]
        # self.act = nn.GELU()                                                 #[128, 216, 16, 25]
        # self.conv2 = nn.Conv1d(num_feat // compress_ratio, num_feat, kernel_size=3, stride=1, padding=1)  #[128, 54, 16, 25]->[128, 216, 16, 25]
        self.att = hyp_temp_attn(num_feat, squeeze_factor)                #[128, 216, 16, 25]->128, 216, 16, 25]
                                 
    def forward(self, x):           #[128, 216, 16, 25]
        NM, C, T, V = x.size()
        x_mod=x.permute(0,3,1,2)
        x_att = self.att(x_mod)     #[128, 25, 216, 16]->[128, 216, 16, 25]
        
        return x_att  
##########-------Hypergraph_Temporal_Attention-------##########



############---------Hypergraph_Convolution---------############
class HGconv(nn.Module):

    def __init__(self,dim_in,dim,pe):
        super().__init__()
        self.pe_proj = nn.Conv2d(dim_in, dim, 1, bias=False)
        self.pe = pe

    def forward(self,x,joint_label,he_weight):
        label = F.one_hot(torch.tensor(joint_label)).float().to(x.device)
        label1=label/label.sum(dim=0,keepdim=True)
        label1=label1/torch.sqrt(label.sum(dim=1,keepdim=True))
        he_weight=he_weight.repeat(25).reshape(25,5)
        label1=he_weight*label1
        label1=label1.permute(1,0)
        norm_label=(label/label.sum(dim=1,keepdim=True))
        h_adj=torch.matmul(norm_label,label1)
        z=x@h_adj
        z = self.pe_proj(z).permute(3, 0, 1, 2)
        e=z.permute(1,2,3,0)

        return e
############---------Hypergraph_Convolution---------############



###########--------Hypergraph_Attention(HGattn)--------###########
class HGattn(nn.Module):

    def __init__(self,dim_in,dim,pe,timesteps):
        super().__init__()
        self.timesteps=timesteps
        self.hyp_attn_temp=hyp_temp_attn_process(timesteps)
        self.hgc1=HGconv(dim_in,dim,pe)
        self.coeff=0.4

    def forward(self,x,joint_label,he_weight):
        op=self.hgc1(x,joint_label,he_weight)
        op_tr=op.permute(0,1,3,2)
        op_attn=self.hyp_attn_temp(op_tr)
        op_final=op_tr+self.coeff*op_attn
        op_final=op_final.permute(0,1,3,2)

        return op_final
###########--------Hypergraph_Attention(HGattn)--------############
    


class unit_vit(nn.Module):

    def __init__(self, dim_in, dim, A, num_of_heads, 
                       add_skip_connection=True,  
                       qkv_bias=False, qk_scale=None, 
                       drop=0., attn_drop=0.,drop_path=0, 
                       act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                       layer=0, insert_cls_layer=0, pe=False, 
                       num_point=25,temp=False,timesteps=64, **kwargs):
        super().__init__()
        
        ##Arguments
        self.dim_in = dim_in
        self.dim = dim
        self.add_skip_connection = add_skip_connection
        self.num_point = num_point
        self.pe = pe
        
        ##Methods
        self.norm1 = norm_layer(dim_in)
        self.attn = MHSA(dim_in, dim, A, num_heads=num_of_heads, 
                         qkv_bias=qkv_bias, qk_scale=qk_scale, 
                         attn_drop=attn_drop, proj_drop=drop, 
                         insert_cls_layer=insert_cls_layer, pe=pe, 
                         num_point=num_point, layer=layer, **kwargs)     
        
        self.pe_proj = nn.Conv2d(dim_in, dim, 1, bias=False)
        if self.dim_in != self.dim:
            self.skip_proj = nn.Conv2d(dim_in, dim, (1, 1), padding=(0, 0), bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if(temp==True):
            self.hgc=HGattn(dim_in,dim,pe,timesteps)
        else:
            self.hgc=HGconv(dim_in,dim,pe)

    def forward(self, x, joint_label, groups, he_weight):
 
        e=self.hgc(x,joint_label,he_weight)
        if self.add_skip_connection:
            if self.dim_in != self.dim:
                x = self.skip_proj(x) + self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))
        else:
            x = self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))

        return x


###########%%%%%%%%%%%%%%%%%-------------VIT-------------%%%%%%%%%%%%%%%%%###########



#############--------Spatio-Temporal_Hypergraph_Transformer(ST-HT)--------#############
class ST_HT(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, num_of_heads=6, residual=True, kernel_size=5, dilations=[1,2], pe=False, num_point=25, layer=0,temp=False,timesteps=64):
        super(ST_HT, self).__init__()
        
        ##Arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        ##Methods
        self.vit1 = unit_vit(in_channels, out_channels, A ,add_skip_connection=residual, num_of_heads=num_of_heads, pe=pe, num_point=num_point, layer=layer,temp=temp,timesteps=timesteps)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, residual=False)
        self.act = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, joint_label, groups,he_weight):
        y = self.act(self.tcn1(self.vit1(x, joint_label, groups,he_weight)) + self.residual(x))
        return y
#############--------Spatio-Temporal_Hypergraph_Transformer(ST-HT)--------#############



##########------Spatio-Temporal_Attentive_Hypergraph_Transformer(STA-HT)------##########
class STA_HT(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, num_of_heads=6, residual=True, kernel_size=5, dilations=[1,2], pe=False, num_point=25, layer=0,temp=False,timesteps=64):
        super(STA_HT, self).__init__()
        
        ##Arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        ##Methods
        self.vit1 = unit_vit(in_channels, out_channels, A ,add_skip_connection=residual, num_of_heads=num_of_heads, pe=pe, num_point=num_point, layer=layer,temp=temp,timesteps=timesteps)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, residual=False)
        self.act = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, joint_label, groups,he_weight):
        y = self.act(self.tcn1(self.vit1(x, joint_label, groups,he_weight)) + self.residual(x))

        return y
##########------Spatio-Temporal_Attentive_Hypergraph_Transformer(STA-HT)------##########



###########-------Frame_Attentive_Hypergraph_Transformer(FAHT)-------###########
class FAHT(nn.Module):

    def __init__(self,A, num_class=60, num_point=20, num_person=2,in_channels=3, drop_out=0,timesteps=64 ,num_of_heads=9,layer=1,  **kwargs):
        super(FAHT, self).__init__()
        self.l1 = ST_HT(3, 24*num_of_heads, A, residual=True,stride=1, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=layer,temp=False)
        self.l2 = STA_HT(24*num_of_heads, 24*num_of_heads, A, residual=True,stride=1, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=layer+1,temp=True,timesteps=timesteps)
        if(layer!=1):
            self.l1 = ST_HT(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=layer,temp=False)
        if(layer==5):
            self.l1 = ST_HT(24*num_of_heads, 24*num_of_heads, A, residual=True,stride=2, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=layer,temp=False)
        if(layer==7):
            self.l2 = STA_HT(24*num_of_heads, 24*num_of_heads, A, residual=True,stride=2, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=layer+1,temp=True,timesteps=timesteps)
        

    def forward(self, x,joint_label,groups,he_weight):
        x = self.l1(x, joint_label, groups,he_weight)
        x = self.l2(x, joint_label, groups,he_weight)
        return x
###########-------Frame_Attentive_Hypergraph_Transformer(FAHT)-------###########



################-----------Hypergraph_Encoder(HypEnc)-----------################
class Hypergraph_Encoder(nn.Module):

    def __init__(self, num_class=60, num_point=20, num_person=2, graph=None, graph_args=dict(), in_channels=3, drop_out=0, num_of_heads=9,  **kwargs):
        super(Hypergraph_Encoder, self).__init__()
        
        ##Arguments
        self.num_of_heads = num_of_heads
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        
        ##Methods
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        A = self.graph.A  # 3,25,25
        self.faht1=FAHT(A, num_class=num_class,num_point=num_point , num_of_heads=num_of_heads, pe=True, timesteps=64, layer=1)
        self.faht2=FAHT(A, num_class=num_class,num_point=num_point , num_of_heads=num_of_heads, pe=True, timesteps=64, layer=3)
        self.faht3=FAHT(A, num_class=num_class,num_point=num_point , num_of_heads=num_of_heads, pe=True, timesteps=32 ,layer=5)
        self.faht4=FAHT(A, num_class=num_class,num_point=num_point , num_of_heads=num_of_heads, pe=True, timesteps=32 ,layer=7)
        self.faht5=FAHT(A, num_class=num_class,num_point=num_point , num_of_heads=num_of_heads, pe=True, timesteps=16 ,layer=9)
        
        #BatchNorm
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        bn_init(self.data_bn, 1)
        
        #FC_Layer
        self.fc = nn.Linear(24*num_of_heads, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        
        #DropOut
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
            

    def forward(self, x,joint_label,he_weight):
        groups = []
        for num in range(max(joint_label)+1):
            groups.append([ind for ind, element in enumerate(joint_label) if element==num])

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)

        ## n, c, t, v
        x = x.view(N, M, V, C, T).contiguous().view(N * M, V, C, T).permute(0, 2, 3, 1)
        x = self.faht1(x, joint_label, groups,he_weight)
        x = self.faht2(x, joint_label, groups,he_weight)
        x = self.faht3(x, joint_label, groups,he_weight)
        x = self.faht4(x, joint_label, groups,he_weight)
        x = self.faht5(x, joint_label, groups,he_weight)
        
        return x
################-----------Hypergraph_Encoder(HypEnc)-----------################



##################-----------Hypergraph_Decoder(HypDec)-----------##################
class Hypergraph_Decoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Hypergraph_Decoder, self).__init__()
        self.conv1=GCNConv(216,64, cached=True) # cached only for transductive learning
        self.conv2=GCNConv(64, 8, cached=True)
        self.conv3=GCNConv(8, 3, cached=True)
        # self.conv4=GCNConv(32, 16, cached=True)
        # self.conv5=GCNConv(16, 8, cached=True)
        # self.conv6=GCNConv(8, 3, cached=True)
        self.bn1=nn.BatchNorm1d(64)
        self.bn2=nn.BatchNorm1d(8)
        # self.bn3=nn.BatchNorm1d(3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x=x.squeeze(0)
        x=x.permute(0,2,1)
        x=self.bn1(x)
        x=x.unsqueeze(3)
        x=x.permute(3,0,2,1)
        x1=self.conv2(x, edge_index)
        x2=x1.relu()
        x2=x2.squeeze(0)
        x2=x2.permute(0,2,1)
        x2=self.bn2(x2)
        x2=x2.unsqueeze(3)
        x2=x2.permute(3,0,2,1)
        
        # x1=self.conv3(x, edge_index).relu()
        # x=self.conv4(x, edge_index).relu()
        # x1=self.conv5(x, edge_index).relu()
        
        return self.conv3(x2, edge_index), x1
##################-----------Hypergraph_Decoder(HypDec)-----------##################



################----------Hyperedge_Attention_Network(HAN)----------################
class Attention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=5):
        super(Attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)                                        
        self.lin1 = nn.Linear(num_feat, num_feat // squeeze_factor, bias=True)     
        self.act = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(num_feat // squeeze_factor, num_feat, bias=True)    
        self.sig = nn.Sigmoid()

    def forward(self, x):                       #[128, 25, 216, 16]
        x_mod = self.pool(x)                    #[128, 25, 216, 16]->[128, 25, 1, 1]
        x_mod = x_mod.permute(0, 2, 3, 1)       #[128, 25, 1, 1]->[128, 1, 1, 25]
        x_mod = self.lin1(x_mod)                #[128, 1, 1, 25]->[128, 1, 1, 5]
        x_mod = self.act(x_mod)                 #[128, 1, 1, 5]->[128, 1, 1, 5]
        x_mod = self.lin2(x_mod)                #[128, 1, 1, 5]->[128, 1, 1, 25]
        x_mod = self.sig(x_mod)                 #[128, 1, 1, 25]->[128, 1, 1, 25]
        return x.permute(0, 2, 3, 1) * x_mod    #[128, 216, 16, 25]*[128, 1, 1, 25]->[128, 216, 16, 25]

class Hyperedge_Attention(nn.Module):

    def __init__(self, num_feat, compress_ratio=4, squeeze_factor=16):
        super(Hyperedge_Attention, self).__init__()

        self.conv1 = nn.Conv1d(num_feat, num_feat // compress_ratio, kernel_size=3, stride=1, padding=1)  
        self.act = nn.GELU()                                                                              
        self.conv2 = nn.Conv1d(num_feat // compress_ratio, num_feat, kernel_size=3, stride=1, padding=1)  
        self.att = Attention(num_feat, squeeze_factor)                                                    
                                 
    def forward(self, x):                                                           #[128, 216, 16, 25]
        NM, C, T, V = x.size()
        x_mod = x.reshape(NM, -1, V).permute(0, 2, 1).contiguous()                  #[128, 216, 16, 25]->[128, 25, 3456]
        x_mod = self.conv1(x_mod)                                                   #[128, 25, 3456]->#[128, 6, 3456]
        x_mod = self.act(x_mod)                                                     #[128, 6, 3456]
        x_mod = self.conv2(x_mod)                                                   #[128, 6, 3456]->[128, 25, 3456]
        x_mod = F.normalize(x_mod.reshape(NM, -1, C, T).contiguous(), p=2, dim=1)   #[128, 25, 3456]->[128, 25, 216, 16]
        x_att = self.att(x_mod)                                                     #[128, 25, 216, 16]->[128, 216, 16, 25]
        return x_att      
################----------Hyperedge_Attention_Network(HAN)----------################



##################-----------Ad_HGformer-----------##################
class Ad_HGformer(nn.Module):

    def __init__(self,num_class,num_features,out_channels,n_points,n_person, graph=None, graph_args=dict(),k=5,drop_out=0):
        super(Ad_HGformer,self).__init__()

        #Arguments
        self.k=k
        self.n_points=n_points
        self.num_class = num_class
        self.num_point = n_points
        self.num_person = n_person

        #Methods
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        self.classifier_model = Hypergraph_Encoder(num_class=num_class,num_person=n_person,num_point=n_points,in_channels=out_channels,graph=graph,graph_args=graph_args)
        self.HAN = Hyperedge_Attention(n_points)
        self.HypDec = Hypergraph_Decoder(in_channels=num_features,out_channels=out_channels)    
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
        self.fc = nn.Linear(216, num_class)
        self.pool = nn.AdaptiveAvgPool2d(1)
    

    ##################-----------Preprocess_Decoder-----------##################
    def Preprocess_Dec(self, channel_x, emb): 
        n_weight=self.pool(F.normalize(channel_x.permute(0,3,1,2), p=2, dim=1)).permute(1,0,2,3) #[128, 1, 1, 25]->[25, 128, 1, 1]
        n_weight=torch.squeeze(n_weight,3)
        n_weight=F.avg_pool2d(n_weight, kernel_size=n_weight.size()[1:3])
        n_weight=torch.squeeze(n_weight,2)
        emb2=emb.permute(0,3,1,2)
        N_, V_, C_, T_ = emb2.shape
        emb2 = emb2.reshape(-1, V_*C_, T_)
        emb2=F.avg_pool1d(emb2, kernel_size=emb2.size()[2])
        emb2 = emb2.reshape(N_, V_, C_, -1).permute(3, 0, 1, 2).contiguous()

        return n_weight, emb2
    ##################-----------Preprocess_Decoder-----------##################


    ##############--------Attentive_Hypergraph__Generator--------##############
    def AttentiveHG_Gen(self, k_emb, n_weight, jl, tr):                                 #[1, 128, 25, 8], [128, 25, 1]
        emb1 = k_emb.detach().permute(2,3,0,1)
        emb1 = F.avg_pool2d(emb1, kernel_size=emb1.size()[2:])
        emb1 = torch.squeeze(emb1, (2,3))
        emb1=F.normalize(emb1, p=2, dim=1)
        X=emb1.cpu().numpy()
        if(tr==True):
            clus=KMeans(n_clusters=self.k, random_state=0, n_init="auto").fit(X)
            joint_label=clus.labels_.tolist()
        else:
            joint_label=jl
        label1 = F.one_hot(torch.tensor(joint_label)).float().to(k_emb.device)
        label1=label1.permute(1,0)
        he_weight=torch.matmul(label1,n_weight)/n_weight.sum()
        he_weight=he_weight.squeeze(1)               
        
        return joint_label, he_weight
    ##############--------Attentive_Hypergraph__Generator--------##############
    

    ##################-----------Process_Input-----------##################
    def Process_Input(x):                           
        n,c,t,v,m=x.size()
        inp=x.permute(0,4,3,1,2)
        inp=torch.reshape(inp,(n*m,v,c,t))
        inp = inp.reshape(-1, v*c, t)
        inp = F.avg_pool1d(inp, kernel_size=inp.size()[2])
        inp = inp.reshape(n*m, v, c, -1).permute(3, 0, 1, 2).contiguous()          
        
        return inp
    ##################-----------Process_Input-----------##################
    

    def forward(self,x,y,tr=True,jl=None,he=None):
        N, C, T, V, M = x.size()

        ####----Edge_Index----####
        adj=self.graph.adj
        adj_t = torch.tensor(adj)
        edge_index = adj_t.nonzero().t().contiguous().to(x.device)

        ####----Classifier----####
        emb = self.classifier_model(x,joint_label=jl,he_weight=he)  #[128, 216, 16, 25]

        ####----Attention----####
        channel_x = self.HAN(emb)                                   #[128, 216, 16, 25]->[128, 216, 16, 25]
        emb=emb + 0.2*channel_x                                     #[128, 216, 16, 25]+[128, 216, 16, 25]->[128, 216, 16, 25]

        n_weight, emb2 = self.Preprocess_Dec(channel_x, emb)
        recon,k_emb=self.HypDec(emb2, edge_index)
        joint_label, he_weight = self.AttentiveHG_Gen(k_emb, n_weight, jl, tr)
        inp= self.Process_Input(x)

        #Classification Head
        _ , C, T, V = emb.size()
        op = emb.view(N, M, C, -1)
        op= op.mean(3).mean(1)
        op = self.drop_out(op)
        op = self.fc(op)
        return op, inp, recon, joint_label, he_weight
##################-----------Ad_HGformer-----------##################