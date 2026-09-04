import torch
import torch.nn as nn


class BurgersRF(nn.Module):
    def __init__(
        self,
        Nx,
        Nt,
        sigma_x=1.0,
        sigma_t=1.0,
        df_x=1,
        df_t=1,
        x_dist="Gaussian",
        t_dist="Gaussian",
        device=None,
    ):
        super(BurgersRF, self).__init__()

        if x_dist == "Gaussian":
            self.Wx = nn.Parameter(torch.randn(1, Nx) / sigma_x, requires_grad=False).to(device)
        elif x_dist == "Student":
            dist = torch.distributions.StudentT(df=df_x, loc=0.0, scale=sigma_x)
            self.Wx = nn.Parameter(dist.sample((1, Nx)), requires_grad=False).to(device)

        if t_dist == "Gaussian":
            self.Wt = nn.Parameter(torch.randn(1, Nt) / sigma_t, requires_grad=False).to(device)
        elif t_dist == "Student":
            dist = torch.distributions.StudentT(df=df_t, loc=0.0, scale=sigma_t)
            self.Wt = nn.Parameter(dist.sample((1, Nt)), requires_grad=False).to(device)

        # bias
        self.bx = nn.Parameter(torch.rand(Nx) * 2 * torch.pi, requires_grad=False).to(device)
        self.bt = nn.Parameter(torch.rand(Nt) * 2 * torch.pi, requires_grad=False).to(device)

        self.model = nn.Sequential(nn.Linear(Nx * Nt, 1, bias=False))

    def forward(self, x, t):
        phi_x = torch.cos(x @ self.Wx + self.bx).to(x.device)
        phi_t = torch.cos(t @ self.Wt + self.bt).to(t.device)

        result_einsum = torch.einsum("bi,bj->bij", phi_x, phi_t).reshape(
            phi_x.shape[0],
            phi_x.shape[1] * phi_t.shape[1],
        ).to(x.device)

        u = self.model(result_einsum)
        return u
