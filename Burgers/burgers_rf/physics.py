import torch

from .config import IC_BC_WEIGHT


def burgers_residual(model, x, t, nv):
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return u_t + u * u_x - nv * u_xx


def boundary_residual(model, m_bd, device):
    t_bd = torch.rand(m_bd, 1).to(device)
    x_left = -torch.ones(t_bd.shape).to(device)
    x_right = torch.ones(t_bd.shape).to(device)
    u_bd1 = model(x_left, t_bd)
    u_bd2 = model(x_right, t_bd)
    return torch.concat((u_bd1, u_bd2))


def initial_residual(model, m_int, g, device):
    x_int = 2 * torch.rand(m_int, 1) - 1
    x_int = x_int.to(device)
    t_int = torch.zeros(x_int.shape)
    t_int = t_int.to(device)
    u_int = model(x_int, t_int)
    return u_int - g(x_int).reshape(u_int.shape)


def loss_fn(model, x, t, m_bd, m_int, nv, g, device):
    """
    Compute the PINN loss
    """
    residual = burgers_residual(model, x, t, nv)
    residual_bd = boundary_residual(model, m_bd, device)
    residual_int = initial_residual(model, m_int, g, device)

    return torch.mean(residual**2) + torch.Tensor([IC_BC_WEIGHT]).to(device) * (
        torch.mean(residual_int**2) + torch.mean(residual_bd**2)
    )


def train(model, optimizer, x, t, m_bd, m_int, nv, g, device, epochs=1000):
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, x, t, m_bd, m_int, nv, g, device)
        loss.backward(retain_graph=True)
        optimizer.step()
        losses.append(loss.item())
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return losses
