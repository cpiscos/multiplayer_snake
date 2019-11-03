import numpy as np
import torch


def compute_gae(next_value, rewards, values, terminals, gamma=0.95, tau=0.95):
    with torch.no_grad():
        values = values + [next_value]
        gae = torch.zeros(values[0].shape)
        returns = []
        for step in reversed(range(len(rewards))):
            gae[terminals[step]] = 0
            next_step_value = values[step + 1]
            next_step_value[terminals[step]] = 0

            delta = rewards[step] + gamma * next_step_value - values[step]
            gae = delta + gamma * tau * gae
            returns.insert(0, (gae + values[step]))
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage, invalids):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids], \
              invalids[rand_ids]


def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, invalids, model, opt, clip_param=0.2):
    losses = []
    actor_losses = []
    critic_losses = []
    entropies = []
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage, invalid in ppo_iter(mini_batch_size, states, actions,
                                                                                  log_probs,
                                                                                  returns, advantages, invalids):
            state, action, old_log_probs, return_, advantage, invalid = state.cuda(), action.cuda(), old_log_probs.cuda(), return_.cuda(), advantage.cuda(), invalid.cuda()
            dist, value = model(state, invalid)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
    return np.mean(losses), np.mean(actor_losses), np.mean(critic_losses), np.mean(entropies)