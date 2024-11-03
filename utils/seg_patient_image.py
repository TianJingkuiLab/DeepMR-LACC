import torch


def seg_patient_image(model, image):             

    seg_mask_batch = []

    for ibatch in range(image.shape[0]):

        seg_mask = []

        for idx in range(image.shape[1]):

            model.eval()
            with torch.no_grad():
                mask = model(image[ibatch][idx].unsqueeze(0).unsqueeze(0))
                mask = (torch.sigmoid(mask) > 0.5).squeeze().long()
                seg_mask.append(mask)

        seg_mask = torch.stack(seg_mask)
        seg_mask_batch.append(seg_mask)

    seg_mask_batch = torch.stack(seg_mask_batch)

    return seg_mask_batch