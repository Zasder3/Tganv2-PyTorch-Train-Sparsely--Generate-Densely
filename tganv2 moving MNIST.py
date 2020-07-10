import torch
import torch.nn.functional as F
import numpy as np
from skvideo import io
from MovingMNIST import MovingMNIST
from models.tganv2_gen import Generator_CLSTM
from models.tganv2_dis import DisMultiResNet
from tqdm.gui import tqdm


def genSamples(g, n=8, e=1):
    with torch.no_grad():
        s = g(torch.rand((n**2, 256), device='cuda')*2-1,
              test=True).cpu().detach().numpy()
    out = np.zeros((1, 20, 64*n, 64*n))

    for j in range(n):
        for k in range(n):
            out[:, :, 64*j:64*(j+1), 64*k:64*(k+1)] = s[j*n + k, 0, :, :, :]

    out = out.transpose((1, 2, 3, 0))
    out = (np.concatenate([out, out, out], axis=3)+1) / 2 * 255
    io.vwrite(f'tganv2moving/gensamples_id{e}.gif', out)


def subsample_real(h, frames=4):
    h = h[:, np.random.randint(min(frames, h.shape[1]))::frames]
    return h


def full_subsample_real(h, frames=4):
    out = []
    for i in range(4):
        if i:
            out.append(subsample_real(out[i-1], frames=frames))
        else:
            out.append(h)

    for i in range(4):
        for j in range(3-i):
            out[i] = F.avg_pool3d(out[i], kernel_size=(1, 2, 2))
    return out


def zero_centered_gp(real_data, pr):
    gradients = torch.autograd.grad(outputs=pr, inputs=real_data,
                                    grad_outputs=torch.ones_like(pr),
                                    create_graph=True, retain_graph=True)

    return sum([torch.sum(torch.square(g)) for g in gradients])


def train():
    epochs = 10000
    batch_size = 32
    lambda_val = 0.5

    # data
    test = MovingMNIST('moving/', train=False)
    loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True,
                                         drop_last=True)

    def dataGen():
        while True:
            for d in loader:
                yield d

    dg = dataGen()
    # gen model
    dis = DisMultiResNet(channels=[32, 32, 64, 128, 256], colors=1).cuda()
    gen = Generator_CLSTM(
        tempc=256,
        zt_dim=4,
        upchannels=[128],
        subchannels=[64, 32, 32],
        n_frames=20,
        colors=1
    ).cuda()
    # original paper doesn't use TTUR but it is generally effective
    disOpt = torch.optim.Adam(dis.parameters(), lr=5e-5, betas=(0, 0.9))
    genOpt = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))

    # train
    # note on loss function: within the current github repo they
    # employ softplus linear loss, if the normal cross entropy
    # is desired one may simply change the comments
    for epoch in tqdm(range(epochs)):
        # discriminator
        disOpt.zero_grad()
        real = torch.cat(next(dg), dim=1).cuda().unsqueeze(2)
        real = real.to(dtype=torch.float32) / 255 * 2 - 1
        real = full_subsample_real(real)
        for i in real:
            i.requires_grad = True
        pr = dis(real)
        dis_loss = zero_centered_gp(real, pr) * lambda_val
        with torch.no_grad():
            fake = gen(torch.rand((batch_size, 256), device='cuda')*2-1)
        pf = dis(fake)
        # dis_loss = -torch.mean(torch.log(pr) + torch.log(1-pf))
        dis_loss += torch.mean(F.softplus(-pr)) + torch.mean(F.softplus(pf))
        dis_loss.backward()
        disOpt.step()
        # generator
        genOpt.zero_grad()
        fake = gen(torch.rand((batch_size, 256), device='cuda')*2-1)
        pf = dis(fake)
        # gen_loss = -torch.mean(torch.log(pf))
        gen_loss = torch.mean(F.softplus(-pf))
        gen_loss.backward()
        genOpt.step()
        # log results
        print('Epoch', epoch, 'Dis', dis_loss.item(), 'Gen', gen_loss.item())
        if epoch % 100 == 0:
            genSamples(gen, e=epoch)


if __name__ == '__main__':
    train()
