import torch
import torch.nn as nn

from .local_features import GaussianLocal, GaussianSumLocal


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
        use_local=False,
        local_type="gaussian",
        local_center=0.0,
        local_width=0.05,
        local_widths=None,
        device=None,
    ):
        super(BurgersRF, self).__init__()
        self.use_local = use_local
        self.local_type = local_type

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

        if self.use_local:
            if local_type == "gaussian":
                self.local_feature = GaussianLocal(center=local_center, width=local_width)
                self.local_coefficients = nn.Parameter(torch.zeros(Nt, 1))
            elif local_type == "gaussian_sum":
                if not local_widths:
                    raise ValueError("local_widths must be provided for local_type='gaussian_sum'")
                self.local_feature = GaussianSumLocal(center=local_center, widths=local_widths)
                self.local_coefficients = nn.Parameter(torch.zeros(Nt, len(local_widths)))
            else:
                raise ValueError(f"Unsupported local feature type: {local_type}")

    def forward(self, x, t):
        phi_x = torch.cos(x @ self.Wx + self.bx).to(x.device)
        phi_t = torch.cos(t @ self.Wt + self.bt).to(t.device)

        result_einsum = torch.einsum("bi,bj->bij", phi_x, phi_t).reshape(
            phi_x.shape[0],
            phi_x.shape[1] * phi_t.shape[1],
        ).to(x.device)

        u_rf = self.model(result_einsum)
        if self.use_local:
            if self.local_type == "gaussian":
                u_local = self.local_feature(x) * (phi_t @ self.local_coefficients)
            else:  # gaussian_sum
                basis = self.local_feature(x)
                temporal = phi_t @ self.local_coefficients
                u_local = (basis * temporal).sum(dim=1, keepdim=True)
            return u_rf + u_local
        return u_rf
