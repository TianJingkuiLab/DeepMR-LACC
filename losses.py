import torch


def cox_loss(y_true, y_pred):

    time_value = y_true[:, 0]                      
    event = y_true[:, 1].bool()                    
    score = y_pred.squeeze()                      

    ix = torch.where(event)[0]                    

    if len(ix) == 0:
        return torch.tensor(0.0)

    sel_mat = (time_value[ix].view(-1, 1) <= time_value).float()

    p_lik = score[ix] - torch.log(torch.sum(sel_mat * torch.exp(score)))

    loss = -torch.mean(p_lik)

    return loss






