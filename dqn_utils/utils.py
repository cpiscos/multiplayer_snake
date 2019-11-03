import torch
import numpy as np


def eps_greedy(adv, invalid, eps=0.01):
    if isinstance(invalid, np.ndarray):
        invalid = torch.BoolTensor(invalid)
    adv = adv.clone()
    adv[invalid] = -100
    opt_action = torch.argmax(adv, 1)
    if eps > 0:
        random_idx = torch.multinomial(torch.tensor([1 - eps, eps]).repeat(len(opt_action), 1), 1).squeeze(1).bool()
        random_action = torch.multinomial((1 - invalid[random_idx].float()), 1).squeeze(1)
        opt_action[random_idx] = random_action.to(opt_action.device)
    return opt_action


def calculate_iqn_loss(model, target_model, states, actions, returns_, next_states, next_invalids, next_terminals, gamma, n_step):
    # states = torch.FloatTensor(states).cuda()
    # actions = torch.LongTensor(actions).cuda()
    # returns_ = torch.FloatTensor(returns_).cuda()
    # next_states = torch.FloatTensor(next_states).cuda()
    # next_invalids = torch.BoolTensor(next_invalids).cuda()
    # next_terminals = torch.BoolTensor(next_terminals).cuda()

    states = states.cuda()
    actions = actions.cuda()
    returns_ = returns_.cuda()
    next_states = next_states.cuda()
    next_invalids = next_invalids.cuda()
    next_terminals = next_terminals.cuda()
    with torch.no_grad():
        next_q_vals, _ = model(next_states, n_tau_samples=32)
        next_q_vals_target, _ = target_model(next_states, n_tau_samples=32)
        best_next_action = eps_greedy(next_q_vals.mean(1), next_invalids, eps=0)

        best_next_q_target = next_q_vals_target[torch.arange(next_q_vals.shape[0]), :, best_next_action]
        best_next_q_target[next_terminals] = 0
        q_target = (returns_.unsqueeze(1) + (gamma ** n_step) * best_next_q_target).unsqueeze(2)
    q_vals, quantiles = model(states, n_tau_samples=32)
    q_vals = q_vals[torch.arange(q_vals.shape[0]), :, actions].unsqueeze(1)

    delta = q_target - q_vals
    t = delta.abs()
    loss = torch.where(t < 1, t.pow(2) / 2, t - 0.5)
    rho_loss = loss * (quantiles - (delta < 0).float()).abs()
    rho_loss = rho_loss.sum(2).mean(1)
    return rho_loss
