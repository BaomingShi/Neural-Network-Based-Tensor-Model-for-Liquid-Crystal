import scipy.io as io
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import random



np.random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matr = io.loadmat('data3DentropyandE2.mat')
data_x = torch.from_numpy(np.transpose(matr["datax"])).float()
data_y = torch.cat((torch.from_numpy(np.transpose(matr["E1_versus_s"])).float(),torch.from_numpy(np.transpose(matr["E2_versus_s"])).float()),dim=1)
data_x = data_x[:,:]


data_y = data_y[:,:]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(2, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 2),
        )


    def forward(self, x):
        x = self.module(x)
        return x



def Loss(net, images, labels):
    criterion = nn.MSELoss()
    logits = net.forward(images)
    loss = criterion(logits, labels)



    return loss

def Train(net, train_images, train_labels, epoch, optim):
    m = 0

    for i in range(epoch):

        loss = Loss(net, train_images, train_labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 200 == 0:
            print('the iteration:%d,the loss:%.3e' % (i + 20, loss.item()))

##Training NNTM
# net = Net()
# optim = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.99))
#
# epoch = 50000
# Train(net, data_x, data_y, epoch, optim)


### Trained NNTM
net = Net()
net=torch.load('net3D_entropyandE2.pth')


net.eval()
net.to(device)

thomson_point = io.loadmat('thomon_4352.mat')
surface_point = torch.from_numpy(thomson_point["X"]).float().T.to(device)




_lambda=torch.tensor([30]).float().to(device)
s = torch.tensor([0.8400]).float().to(device)
kappa=torch.tensor([12]).float().to(device)


def energy(trQ2andtrQ3):

    E = net(trQ2andtrQ3)

    E_entropy = E[:,0]
    E2 = E[:,1]
    return E_entropy+kappa/2*E2


def max_eig(q1, q2, q3, q4, q5):

    q1 = float(q1)
    q2 = float(q2)
    q3 = float(q3)
    q4 = float(q4)
    q5 = float(q5)
    matrix = np.array([[q1, q3, q4], [q3, q2, q5], [q4, q5, -q1-q2]])


    eigenvalues, eigenvectors = np.linalg.eig(matrix)


    max_eigenvalue_index = np.argmax(eigenvalues)
    max_eigenvector = eigenvectors[:, max_eigenvalue_index]
    return torch.tensor(max_eigenvector).to(device), eigenvalues[max_eigenvalue_index]



def plot_quiver(net):
    # quiver
    x_start = -1
    y_start = -1
    z_start = -1
    x_end = 1
    y_end = 1
    z_end = 1
    resolution = 30
    random_ratio = 0.05


    x_list = np.linspace(x_start, x_end, resolution).tolist()
    y_list = np.linspace(y_start, y_end, resolution).tolist()
    z_list = np.linspace(z_start, z_end, resolution).tolist()
    X, Y, Z = np.meshgrid(x_list, y_list, z_list)
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    W = np.zeros_like(X)
    C = np.zeros_like(X)
    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            for k, z in enumerate(z_list):
                if x ** 2 + y ** 2 + z ** 2 > 1:
                    C[i, j, k] = float("inf")
                else:
                    q1 = net.netq1(torch.tensor([X[i, j, k], Y[i, j, k], Z[i, j, k]]).float().to(device))
                    q2 = net.netq2(torch.tensor([X[i, j, k], Y[i, j, k], Z[i, j, k]]).float().to(device))
                    q3 = net.netq3(torch.tensor([X[i, j, k], Y[i, j, k], Z[i, j, k]]).float().to(device))
                    q4 = net.netq4(torch.tensor([X[i, j, k], Y[i, j, k], Z[i, j, k]]).float().to(device))
                    q5 = net.netq5(torch.tensor([X[i, j, k], Y[i, j, k], Z[i, j, k]]).float().to(device))


                    if random.random() < random_ratio:
                        max_eig_vec, maximumeig = max_eig(q1, q2, q3, q4, q5)
                        if maximumeig <= 1e-1:
                            maximumeig = 0
                        U[i, j, k] = max_eig_vec[0]*maximumeig
                        V[i, j, k] = max_eig_vec[1]*maximumeig
                        W[i, j, k] = max_eig_vec[2]*maximumeig

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)

    ax.set_title('Quiver Plot on a Sphere')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def generate_grid_3d(mode, n):
    if mode == 'random':
        phi = np.random.uniform(0, 2*np.pi, n)
        costheta = np.random.uniform(-1, 1, n)
        theta = np.arccos(costheta)
        r = np.random.uniform(0, 1, n) ** (1/3)  
        points = np.column_stack((r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)))
    elif mode == 'uniform':
        step = 2 / (n ** (1/3))
        # 生成均匀分布的点
        points = []
        for i in range(int(n ** (1/3))):
            for j in range(int(n ** (1/3))):
                for k in range(int(n ** (1/3))):
                    x = -1 + i * step + step / 2
                    y = -1 + j * step + step / 2
                    z = -1 + k * step + step / 2
                    if x ** 2 + y ** 2 + z ** 2 <= 1:
                        points.append([x, y, z])
        points = np.array(points)
    else:
        print("Unknown mode.\n")
        points = None
    return points



class solve_net:

    def __init__(self):

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("gpu")

        self.netq1 = nn.Sequential(
            nn.Linear(3, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
            nn.Tanh(),
        ).to(device)

        self.netq2 = nn.Sequential(
            nn.Linear(3, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
            nn.Tanh(),
        ).to(device)


        self.netq3 = nn.Sequential(
            nn.Linear(3, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
            nn.Tanh(),
        ).to(device)

        self.netq4 = nn.Sequential(
            nn.Linear(3, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
            nn.Tanh(),
        ).to(device)

        self.netq5 = nn.Sequential(
            nn.Linear(3, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
            nn.Tanh(),
        ).to(device)
        self.iterpretrain = 1


        central_location = -0.1
        inner_radius = 0.7 # abs(central_location)+inner_radius must lower than 1
        center = torch.tensor([0.0, 0.0, central_location])

        n = 100000
        point_inner = torch.tensor(generate_grid_3d('uniform', n)).float()

        distances = torch.norm(point_inner - center, dim=1)


        X_inside = point_inner[distances > inner_radius]

        number_outterpoint = surface_point.shape[0]
        num_to_select = int(inner_radius**2 * number_outterpoint)
        indices = torch.randperm(number_outterpoint)


        selected_points = surface_point[indices[:num_to_select]]

        normal_vec = selected_points

        surface_point_inner=selected_points*inner_radius
        surface_point_inner[:, 2] += central_location

        point_boundary = surface_point

        # boundary nodes, the last three rows are normal vector
        point_boundary_total = torch.cat((point_boundary.float().clone().detach(), surface_point_inner), dim=0)
        normal_vec_total = torch.cat((point_boundary.float().clone().detach(), normal_vec), dim=0)
        self.X_boundary = torch.cat((point_boundary_total,normal_vec_total),dim=1)



        self.X_inside = X_inside.to(device)
        self.X_boundary = self.X_boundary.to(device)

        self.X_inside.requires_grad = True


        self.criterion = torch.nn.MSELoss()

        self.iter = 1



        self.q1adam = torch.optim.Adam(self.netq1.parameters(),lr=0.0001)
        self.q2adam = torch.optim.Adam(self.netq2.parameters(),lr=0.0001)
        self.q3adam = torch.optim.Adam(self.netq3.parameters(), lr=0.0001)
        self.q4adam = torch.optim.Adam(self.netq4.parameters(), lr=0.0001)
        self.q5adam = torch.optim.Adam(self.netq5.parameters(), lr=0.0001)

        self.Qadam = torch.optim.Adam(
            [{'params': self.netq1.parameters()},
             {'params': self.netq2.parameters()},
             {'params': self.netq3.parameters()},
             {'params': self.netq4.parameters()},
             {'params': self.netq5.parameters()}],
            lr=0.0001
        )
        self.scheduler = StepLR(self.Qadam, step_size=200, gamma=0.97)

    def loss_func(self):

        self.Qadam.zero_grad()

        U_pred_boundaryq1 = self.netq1(self.X_boundary[:,0:3])
        U_pred_boundaryq2 = self.netq2(self.X_boundary[:,0:3])
        U_pred_boundaryq3 = self.netq3(self.X_boundary[:,0:3])
        U_pred_boundaryq4 = self.netq4(self.X_boundary[:,0:3])
        U_pred_boundaryq5 = self.netq5(self.X_boundary[:,0:3])

        stacked_tensor = torch.stack(
            [U_pred_boundaryq1, U_pred_boundaryq3, U_pred_boundaryq4, U_pred_boundaryq3, U_pred_boundaryq2,
             U_pred_boundaryq5, U_pred_boundaryq4, U_pred_boundaryq5, -U_pred_boundaryq1 - U_pred_boundaryq2], dim=1)
        reshaped_tensor = stacked_tensor.view(-1, 3, 3)
        ## Q*\nu
        Qnu = torch.matmul(reshaped_tensor, self.X_boundary[:, 3:].unsqueeze(2)).squeeze(dim=2)

        loss_boundary = self.criterion(
            Qnu, -s / 3 * self.X_boundary[:, 3:])  # boundary loss


        U_insideq1 = self.netq1(self.X_inside)
        U_insideq2 = self.netq2(self.X_inside)
        U_insideq3 = self.netq3(self.X_inside)
        U_insideq4 = self.netq4(self.X_inside)
        U_insideq5 = self.netq5(self.X_inside)


        dq1_dX = torch.autograd.grad(
            inputs=self.X_inside,
            outputs=U_insideq1,
            grad_outputs=torch.ones_like(U_insideq1),
            retain_graph=True,
            create_graph=True
        )[0]
        dq1_dx = dq1_dX[:, 0]
        dq1_dy = dq1_dX[:, 1]
        dq1_dz = dq1_dX[:, 2]

        dq2_dX = torch.autograd.grad(
            inputs=self.X_inside,
            outputs=U_insideq2,
            grad_outputs=torch.ones_like(U_insideq2),
            retain_graph=True,
            create_graph=True
        )[0]
        dq2_dx = dq2_dX[:, 0]
        dq2_dy = dq2_dX[:, 1]
        dq2_dz = dq2_dX[:, 2]

        dq3_dX = torch.autograd.grad(
            inputs=self.X_inside,
            outputs=U_insideq3,
            grad_outputs=torch.ones_like(U_insideq3),
            retain_graph=True,
            create_graph=True
        )[0]
        dq3_dx = dq3_dX[:, 0]
        dq3_dy = dq3_dX[:, 1]
        dq3_dz = dq3_dX[:, 2]

        dq4_dX = torch.autograd.grad(
            inputs=self.X_inside,
            outputs=U_insideq4,
            grad_outputs=torch.ones_like(U_insideq4),
            retain_graph=True,
            create_graph=True
        )[0]
        dq4_dx = dq4_dX[:, 0]
        dq4_dy = dq4_dX[:, 1]
        dq4_dz = dq4_dX[:, 2]

        dq5_dX = torch.autograd.grad(
            inputs=self.X_inside,
            outputs=U_insideq5,
            grad_outputs=torch.ones_like(U_insideq5),
            retain_graph=True,
            create_graph=True
        )[0]
        dq5_dx = dq5_dX[:, 0]
        dq5_dy = dq5_dX[:, 1]
        dq5_dz = dq5_dX[:, 2]

        loss_elastic = torch.dot(dq1_dx, dq1_dx) / len(dq1_dx) + torch.dot(dq1_dy, dq1_dy) / len(dq1_dy) + torch.dot(
            dq1_dz, dq1_dz) / len(dq1_dz) + torch.dot(dq2_dx, dq2_dx) / len(dq2_dx) + torch.dot(dq2_dy, dq2_dy) / len(
            dq2_dy) + torch.dot(dq2_dz, dq2_dz) / len(dq2_dz) + torch.dot(dq1_dx, dq2_dx) / len(dq2_dx) + torch.dot(
            dq1_dy, dq2_dy) / len(dq2_dy) + torch.dot(dq1_dz, dq2_dz) / len(dq2_dz) + torch.dot(dq3_dx, dq3_dx) / len(
            dq3_dx) + torch.dot(dq3_dy, dq3_dy) / len(dq3_dy) + torch.dot(dq3_dz, dq3_dz) / len(dq3_dz) + torch.dot(
            dq4_dx, dq4_dx) / len(dq4_dx) + torch.dot(dq4_dy, dq4_dy) / len(dq4_dy) + torch.dot(dq4_dz, dq4_dz) / len(
            dq4_dz) + torch.dot(dq5_dx, dq5_dx) / len(dq5_dx) + torch.dot(dq5_dy, dq5_dy) / len(dq5_dy) + torch.dot(
            dq5_dz, dq5_dz) / len(dq5_dz)

        stacked_tensor_inner = torch.stack(
            [U_insideq1, U_insideq3, U_insideq4, U_insideq3, U_insideq2,
             U_insideq5, U_insideq4, U_insideq5, -U_insideq1 - U_insideq2], dim=1)
        reshaped_tensor_inner = stacked_tensor_inner.view(-1, 3, 3)

        Q_Q = torch.matmul(reshaped_tensor_inner, reshaped_tensor_inner)  #  Q * Q
        Q_Q_Q = torch.matmul(Q_Q, reshaped_tensor_inner)  #  Q * Q * Q

        # trace
        tr_Q_Q = torch.einsum('...ii->...', Q_Q)  #  tr(Q * Q)
        tr_Q_Q_Q = torch.einsum('...ii->...', Q_Q_Q)  #  tr(Q * Q * Q)
        combined_traces = torch.stack((tr_Q_Q.squeeze(), tr_Q_Q_Q.squeeze()), dim=1)
        loss_bulk = torch.sum(energy(combined_traces)) / len(dq1_dx)

        loss = loss_elastic + _lambda * loss_bulk + 1000 * loss_boundary

        loss.backward()

        if self.iter % 100 == 0:
            print(self.iter, loss.item())
        self.iter = self.iter + 1
        return loss


    def loss_func_pretrain(self):

        self.Qadam.zero_grad()



        U_insideq1 = self.netq1(self.X_inside)
        U_insideq2 = self.netq2(self.X_inside)
        U_insideq3 = self.netq3(self.X_inside)
        U_insideq4 = self.netq4(self.X_inside)
        U_insideq5 = self.netq5(self.X_inside)

        NNN = U_insideq5.shape[0]

        loss_pre = self.criterion(U_insideq1, -s/3*torch.ones(NNN,1).to(device)) + self.criterion(U_insideq2, -s/3*torch.ones(NNN,1).to(device)) + self.criterion(U_insideq3, 0/3*torch.ones(NNN,1).to(device)) + self.criterion(U_insideq4, 0/3*torch.ones(NNN,1).to(device)) + self.criterion(U_insideq5, 0/3*torch.ones(NNN,1).to(device))
        loss_pre.backward()

        if self.iterpretrain % 100 == 0:
            print(self.iterpretrain, loss_pre.item())
        self.iterpretrain = self.iterpretrain + 1
        return loss_pre


    def pre_train(self):
        self.netq1.train()
        self.netq2.train()
        self.netq3.train()
        self.netq4.train()
        self.netq5.train()


        for i in range(500):
            self.Qadam.step(self.loss_func_pretrain)

    def train(self):
        self.netq1.train()
        self.netq2.train()
        self.netq3.train()
        self.netq4.train()
        self.netq5.train()



        for i in range(5000):
            self.Qadam.step(self.loss_func)
            self.scheduler.step()



solve_minimum = solve_net()

solve_minimum.pre_train()
solve_minimum.train()

plot_quiver(solve_minimum)


### better Visualization in visualization_in_matlab.m
point_total=torch.cat((solve_minimum.X_inside[:,0:3],solve_minimum.X_boundary[:,0:3]),0)
q1=solve_minimum.netq1(point_total)
q2=solve_minimum.netq2(point_total)
q3=solve_minimum.netq3(point_total)
q4=solve_minimum.netq4(point_total)
q5=solve_minimum.netq5(point_total)


point_total=point_total.cpu().detach()
q1=q1.cpu().detach()
q2=q2.cpu().detach()
q3=q3.cpu().detach()
q4=q4.cpu().detach()
q5=q5.cpu().detach()
point_total=np.array(point_total)

q1=np.array(q1)
q2=np.array(q2)
q3=np.array(q3)
q4=np.array(q4)
q5=np.array(q5)

mdic={'point_total':point_total,'q1':q1,'q2':q2,'q3':q3,'q4':q4,'q5':q5}
import scipy
scipy.io.savemat("test_shell.mat", mdic)
