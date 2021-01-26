import sklearn.linear_model as linear_model
import copy
class Lasso(nn.Module):
    "Lasso for compressing dictionary"
    def __init__(self, input_size):
        super(Lasso, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)
    def forward(self, x):
        out = self.linear(x)
        return out

def soft_th(X,threshold = 5):
     return np.sign(X) * np.maximum((np.abs(X) - threshold), np.zeros(X.shape))
def torch_soft_operator(X,threshold = 0.1):
    np_X = X.numpy()
    tmp = np.sign(np_X) * np.maximum((np.abs(np_X) - threshold), np.zeros(np_X.shape))
    return torch.from_numpy(tmp.astype(np.float32))
def lasso(x, y, lmbda = 1,lr=0.005, max_iter=2000, tol=1e-4, opt='SGD'):
    # x = x.detach()
    # y = y.detach()
    
    lso = Lasso(x.shape[1])
    criterion = nn.MSELoss(reduction='mean')
    if opt == 'adam':
        optimizer = optim.Adam(lso.parameters(), lr=lr)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(lso.parameters(), lr=lr)
    elif opt == "SGD":
        optimizer = optim.SGD(lso.parameters(), lr=lr)
    w_prev = torch.tensor(0.)
    for it in range(max_iter):
        # lso.linear.zero_grad()
        optimizer.zero_grad()
        out = lso(x)
        fn = criterion(out, y)
        l1_norm = lmbda * torch.norm(lso.linear.weight, p=1)
        l1_crit = nn.L1Loss()
        target = Variable(torch.from_numpy(np.zeros((x.shape[1],1))))

        loss = 0.5*fn + lmbda * F.l1_loss(lso.linear.weight, target=torch.zeros_like(lso.linear.weight.detach()), size_average=False)
        loss.backward()
        optimizer.step()
        # pdb.set_trace()
        if it ==0:
            w = lso.linear.weight.detach()
        else:
            with torch.no_grad():
                sign_w = torch.sign(w)
                ## hard-threshold
#                 lso.linear.weight[torch.where(torch.abs(lso.linear.weight) <= lmbda*lr)] = 0
                ## soft-threshold
#                 z = lso.linear.weight
#                 lso.linear.weight[torch.where(z != 0)] = torch.sign(z[torch.where(z != 0)]) * torch.maximum(torch.abs(z[torch.where(z != 0)]) - lr*lmbda,\
#                                                                                                             torch.zeros_like(z[torch.where(z != 0)]))
                w = copy.deepcopy(lso.linear.weight.detach())
        if it % 500 == 0:
#             print(target.shape)
            print(loss.item(),end =" ")
            print(torch.norm(lso.linear.weight.detach(), p=1),end = " ")
            print("l1_crit: ",l1_crit(lso.linear.weight.detach(),target),end = " ")
            print("F L1: ",F.l1_loss(lso.linear.weight.detach(), target=torch.zeros_like(lso.linear.weight.detach()), size_average=False))
    return lso.linear.weight.detach()



## generate data
np.random.seed(5)
n = 100
p = 10
beta = np.zeros([p]).astype(np.float32)
beta[0] = 10
beta[1] =10
beta[2] = -10
beta[3] =-10
X = np.random.rand(n,p).astype(np.float32)
Y = np.dot(X,beta)
a = torch.from_numpy(X)
b = torch.from_numpy(Y.reshape(-1,1))


## main 
penalty = 0.1
r = lasso(a, b, lmbda = penalty,lr= 0.01,opt='SGD',max_iter = 5000)
print("\n")
print("SGD-L1 clipping")
print(r)
# print("soft-threshold")
# print(soft_th(r.numpy(),threshold = penalty))
l = linear_model.Lasso(alpha=penalty, fit_intercept=False)
l.fit(a, b)
b_hat = l.predict(a) 
# l.path(a, b, verbose=True)
print("sklearn lasso result:")
print(l.coef_)


## SGDRegressor
sgd = linear_model.SGDRegressor(alpha = penalty,penalty = "l1",learning_rate= "constant",eta0 = 0.01,max_iter = 1000,fit_intercept=False)
sgd.fit(a,b)
sgd.coef_