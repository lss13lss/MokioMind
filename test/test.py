import torch
import torch.nn as nn
import math

class MHA(nn.Module):  # 定义多头注意力机制(Multi-Head Attention)类，继承自PyTorch的nn.Module
    def __init__(self, d_model, num_heads, dropout=0.1):  # 初始化函数，参数包括：模型维度、头数、丢弃率
        super().__init__()  # 调用父类nn.Module的初始化函数
        self.d_model = d_model  # 保存模型维度
        self.num_heads = num_heads  # 保存注意力头的数量
        self.head_dim = d_model // num_heads  # 计算每个注意力头的维度

        # 定义四个线性投影层，用于计算查询(Q)、键(K)、值(V)和输出
        self.q_proj = nn.Linear(d_model, d_model)  # 查询投影层
        self.k_proj = nn.Linear(d_model, d_model)  # 键投影层
        self.v_proj = nn.Linear(d_model, d_model)  # 值投影层
        self.out_proj = nn.Linear(d_model, d_model)  # 输出投影层

        self.dropout = nn.Dropout(dropout)  # 定义dropout层，用于防止过拟合

    def forward(self, x, mask=None):  # 前向传播函数，参数：输入x和可选的掩码
        batch_size, seq_len, _ = x.size()  # 获取输入张量的尺寸信息
        q = self.q_proj(x)  # 将输入通过线性层投影得到查询Q
        k = self.k_proj(x)  # 将输入通过线性层投影得到键K
        v = self.v_proj(x)  # 将输入通过线性层投影得到值V

        # 将Q、K、V重塑并转置，以便多头并行计算
        # 形状变化: [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数：Q和K的点积，再除以head_dim的平方根进行缩放
        # 形状: [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 如果提供了掩码，则将掩码为0的位置的注意力分数设置为极小值(-1e9)
        # 这样在softmax后，这些位置的注意力权重几乎为0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)  
        
        # 对注意力分数应用softmax得到注意力概率分布，并应用dropout
        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))
        
        # 将注意力概率与V相乘，得到注意力输出
        # 形状: [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_probs, v)
        
        # 将多头注意力输出重塑回原始形状
        # 形状变化: [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 通过输出投影层得到最终结果
        return self.out_proj(attn_output)
    
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MHA(d_model, num_heads, dropout)  # 多头注意力机制
        self.ffn = FFN(d_model, d_ff, dropout)  # 前馈神经网络
        self.norm1 = nn.LayerNorm(d_model)  # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二个层归一化
        self.dropout = nn.Dropout(dropout)  # dropout层，用于残差连接后的正则化

    def forward(self, x, mask=None):
        # 前归一化架构：先进行层归一化，再进入子模块，最后加上残差连接
        # 多头注意力子层
        residual = x  # 保存输入作为残差连接
        x = self.norm1(x)  # 前归一化：先归一化
        attn_output = self.attn(x, mask)  # 然后通过多头注意力
        x = residual + self.dropout(attn_output)  # 残差连接
        
        # 前馈网络子层
        residual = x  # 保存当前状态作为残差连接
        x = self.norm2(x)  # 前归一化：先归一化
        ffn_output = self.ffn(x)  # 然后通过前馈网络
        x = residual + self.dropout(ffn_output)  # 残差连接
        
        return x