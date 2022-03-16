import torch
from pandas import np
from torch import nn


class RNN(nn.Module):
    def __init__(self ,):
        # 定义
        super(RNN, self).__init__()

        input_size =1  # 输入x的特征数 也就是 feature
        hidden_size =20 # 隐含层的特征数量
        num_layers =2   # rnn的层数

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size =input_size
        self.output_size =5 # 自己设置的参数，self.linear经过全连接之后输出的维度

        self.embedding = nn.Embedding(10 ,input_size ,padding_idx=1)
        #                       词典的大小  每个词嵌入的维度
        # padding_idx (python:int, optional)
        # 填充id，比如输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）


        self.rnn = nn.RNN(self.input_size ,self.hidden_size ,self.num_layers, batch_first = True)
        # batch_first = True 表示 rnn的的输入数据的维度是 [batc，seq_len，input_size]
        # 如果不写 就是默认 也就是 batch_first=False,[seq_len，batch，input_size]
        # 输出的全链接层

        self.linear = nn.Linear(self.hidden_size ,self.output_size)

        # 最后的logsoftmax层
        self.softmax = nn.LogSoftmax()

    def forward(self ,input):

        input =torch.LongTensor(input.numpy()  )# input 直接转化成 LongTensor 失败 所以先转化成 .numpy()，再LongTensor。

        output = self.embedding(input)  # self.embedding = nn.Embedding(10,input_size,padding_idx=1)

        output =output.unsqueeze(0  )  # 在第一个位置增加一个一维度
        # print(output.shape) # ([1, 1, 1])  # x的尺寸：序列长度seq_len,batch_size,input_size

        h_0 =(torch.randn(2 ,1 ,20))
        # h_0尺寸：num_layers * num_directions,batch_size, hidden_size,
        # num_directions是方向数。1是单向 2 是双向 rnn

        output, h_n = self.rnn(output, h_0)
        # output尺寸：序列长度seq_len,batch,hidden_size* num_directions,torch.Size([1, 1, 20])
        # print(h_n.shape) h_n的尺寸=h_0尺寸,

        output = output[: ,-1 ,:] # 这个操作不好解释 可以具体情况具体分析
        # print(output.shape)#torch.Size([1, 20])

        output = self.linear(output) # self.linear = nn.Linear(self.hidden_size,self.output_size)
        # print(output) # torch.Size([1, 5])          # self.hidden_size=20,self.output_size=5

        output = self.softmax(output)
        # print(output.shape) #torch.Size([1, 5])
        return output, h_n
    def initHidden(self):
        # 对隐含单元的初始化
        return torch.zeros(1, self.hidden_size)
    
    
    
# model = RNN()
# criterion = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
# train_set = [[3, 0, 0, 1, 1, 2],
#              [3, 0, 1, 2],
#              [3, 0, 0, 0, 1, 1, 1, 2],
#              [3, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]]



# # 重复进行50次试验
# num_epoch = 5
# loss_list = []
# for epoch in range(num_epoch):
#     train_loss = 0
#     # 对train_set中的数据进行随机洗牌，以保证每个epoch得到的训练顺序都不一样。
#     np.random.shuffle(train_set)
#     # 对train_set中的数据进行循环
#     for i, seq in enumerate(train_set):
#         loss = 0
#         for t in range(len(seq) - 1):

#             x = torch.Tensor([seq[t]])
#             y = torch.Tensor([seq[ t +1]])

#             output, h_n = model(x)  # 综合加了RNN的模型输出 输出
#             loss += criterion(output ,y.long())

#         loss = 1.0 * loss / len(seq)  # 计算每字符的损失数值
#         print(loss)
#         optimizer.zero_grad() # 梯度清空
#         loss.backward()  # 反向传播
#         optimizer.step()  # 一步梯度下降