import torch
import torch.nn as nn
import numpy as np

def calculate_kernel_bandwidth(A):
    IP_0 = 0
    for i in range(A.shape[0]):
        IP = np.square(np.linalg.norm(A[i]))
        # print(IP)
        IP_0 += IP
    lambd = 1/((1/A.shape[0]) * IP_0)
    return lambd

def calculate_GaussianKernel_sim(A):

    kernel_bandwidth = calculate_kernel_bandwidth(A)
    gauss_kernel_sim = np.zeros((A.shape[0],A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            gaussianKernel = np.exp(-kernel_bandwidth * np.square(np.linalg.norm(A[i] - A[j])))
            gauss_kernel_sim[i][j] = gaussianKernel

    return gauss_kernel_sim

class StackedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(StackedAutoencoder, self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            self.encoder.add_module(f'encoder_linear_{i}', nn.Linear(prev_dim, dim))
            self.encoder.add_module(f'encoder_relu_{i}', nn.ReLU(True))
            prev_dim = dim

        self.latent_dim = prev_dim

        for i, dim in enumerate(reversed(hidden_dims)):
            self.decoder.add_module(f'decoder_linear_{i}', nn.Linear(prev_dim, dim))
            self.decoder.add_module(f'decoder_relu_{i}', nn.ReLU(True))
            prev_dim = dim

        self.decoder.add_module('decoder_final', nn.Linear(prev_dim, input_dim))
        self.decoder.add_module('decoder_sigmoid', nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


#dataset1
lnc_fun_dict = np.loadtxt(".../dataset1/lnc_fusion_sim.txt")
disease_model1_dict = np.loadtxt('.../dataset1/dis_fusion_sim.txt')
mi_fun_dict=np.loadtxt(".../dataset1/mi_fusion_sim.txt")
lnc_dis=np.loadtxt(".../dataset1/lnc_dis_association.txt")
mi_dis=np.loadtxt(".../dataset1/mi_dis.txt")
mi_lnc=np.loadtxt('.../dataset1/lnc_mi_interaction.txt').T

lnc_ga_LDA=calculate_GaussianKernel_sim(lnc_dis)
dis_ga_LDA=calculate_GaussianKernel_sim(lnc_dis.T)

mi_ga_MDA=calculate_GaussianKernel_sim(mi_dis)
dis_ga_MDA=calculate_GaussianKernel_sim(mi_dis.T)

lnc_ga_LMI=calculate_GaussianKernel_sim(mi_lnc.T)
mi_ga_LMI=calculate_GaussianKernel_sim(mi_lnc)

hidden_dims = [256, 128, 64]  # 隐藏层维度
num_epochs = 10
batch_size = 16
learning_rate = 1e-3

#dataset2
'''disease_model1_dict = np.loadtxt("/data3/yyzhou/BATNDA/GCLMTP-main/GCLMTP-main/data/dataset2/dataset2/dis_fusion_sim.txt")
lnc_fun_dict = np.loadtxt(".../dataset2/lnc_fusion_sim.txt")
mi_fun_dict = np.loadtxt(".../dataset2/mi_fusion_sim.txt")

di_lnc = pd.read_csv('.../dataset2/di_lnc_intersection.csv', index_col='Unnamed: 0')
di_mi = pd.read_csv('.../dataset2/di_mi_intersection.csv', index_col='Unnamed: 0')
mi_lnc = pd.read_csv('.../dataset2/mi_lnc_intersection.csv', index_col='Unnamed: 0')

lnc_dis = di_lnc.values.T
mi_dis = di_mi.values.T
mi_lnc = mi_lnc.values

lnc_ga_LDA=calculate_GaussianKernel_sim(lnc_dis)
dis_ga_LDA=calculate_GaussianKernel_sim(lnc_dis.T)

mi_ga_MDA=calculate_GaussianKernel_sim(mi_dis)
dis_ga_MDA=calculate_GaussianKernel_sim(mi_dis.T)

lnc_ga_LMI=calculate_GaussianKernel_sim(mi_lnc.T)
mi_ga_LMI=calculate_GaussianKernel_sim(mi_lnc)
'''

# 数据准备
#X = lnc_ga_LDA # 输入数据,形状为(num_samples, input_dim)
#X=dis_ga_LDA
#X=disease_model1_dict
X=lnc_fun_dict
#X=lnc_dis
#X=lnc_dis.T
#X=mi_ga_MDA
#X=dis_ga_MDA
#X=mi_fun_dict
#X=mi_dis
#X=mi_dis.T
#X=lnc_ga_LMI
#X=mi_ga_LMI
#X=mi_lnc
#X=mi_lnc.T


input_dim=X.shape[1]
X = torch.tensor(X, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(X)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
model = StackedAutoencoder(input_dim, hidden_dims)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data[0]

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    encoded_features = model.encoder(torch.tensor(X, dtype=torch.float32)).numpy()
    encoded_features = torch.tensor(encoded_features)

    torch.save(encoded_features,
               ".../processed_data/dataset1/lnc_fun.npy")
