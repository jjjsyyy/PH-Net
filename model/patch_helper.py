import random
import numpy as np
import torch
from torch.nn import functional as F


def devide_into_patchs(image, label, logit, pred, patch_num):
    assert image.shape[-1] % patch_num == 0
    assert label.shape[-1] % patch_num == 0

    image = image.unsqueeze(1)
    image1 = torch.chunk(image, int(patch_num), dim=-1)
    image1 = torch.cat(image1, dim=1)
    image1 = image1.unsqueeze(1)
    image2 = torch.chunk(image1, int(patch_num), dim=-2)
    image2 = torch.cat(image2, dim=1)

    label = label.unsqueeze(1)
    label1 = torch.chunk(label, int(patch_num), dim=-1)
    label1 = torch.cat(label1, dim=1)
    label1 = label1.unsqueeze(1)
    label2 = torch.chunk(label1, int(patch_num), dim=-2)
    label2 = torch.cat(label2, dim=1)

    logit = logit.unsqueeze(1)
    logit1 = torch.chunk(logit, int(patch_num), dim=-1)
    logit1 = torch.cat(logit1, dim=1)
    logit1 = logit1.unsqueeze(1)
    logit2 = torch.chunk(logit1, int(patch_num), dim=-2)
    logit2 = torch.cat(logit2, dim=1)

    pred = pred.unsqueeze(1)
    pred1 = torch.chunk(pred, int(patch_num), dim=-1)
    pred1 = torch.cat(pred1, dim=1)
    pred1 = pred1.unsqueeze(1)
    pred2 = torch.chunk(pred1, int(patch_num), dim=-2)
    pred2 = torch.cat(pred2, dim=1)

    return image2, label2, logit2, pred2


def compute_hard_ratio(pred):
    with torch.no_grad():
        # prob = torch.softmax(pred, dim=-3)
        prob = pred.clone()
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=-3)
        hard_ratio = entropy / torch.log(torch.tensor(2.))
        hard_ratio = hard_ratio.mean(dim=[-2, -1])

    return hard_ratio


def generate_patch_data(image, label, logit, hard_ratio, ex_ratio=40, alpha=30, mix_type="easy_easy"):
    assert mix_type in ["easy_easy", "easy_hard", "easy_rand"]
    batch_size, patch_num, _, _, im_h, im_w = image.shape
    easy_mask_list = []
    hard_mask_list = []
    rand_mask_list = []

    easy_image_list = []
    easy_label_list = []
    easy_logit_list = []
    hard_image_list = []
    hard_label_list = []
    hard_logit_list = []
    rand_image_list = []
    rand_label_list = []
    rand_logit_list = []

    for i in range(batch_size):
        easy_mask, hard_mask, rand_mask = generate_patch_mask(hard_ratio[i], ex_ratio, alpha)
        easy_mask_list.append(easy_mask)
        hard_mask_list.append(hard_mask)
        rand_mask_list.append(rand_mask)

        easy_image_list.append(image[i][easy_mask])
        easy_label_list.append(label[i][easy_mask])
        easy_logit_list.append(logit[i][easy_mask])

        hard_image_list.append(image[i][hard_mask])
        hard_label_list.append(label[i][hard_mask])
        hard_logit_list.append(logit[i][hard_mask])

        rand_image_list.append(image[i][rand_mask])
        rand_label_list.append(label[i][rand_mask])
        rand_logit_list.append(logit[i][rand_mask])

    for i in range(batch_size):

        if mix_type == "easy_easy":
            image[i][easy_mask_list[i]] = easy_image_list[(i + 1) % batch_size]
            label[i][easy_mask_list[i]] = easy_label_list[(i + 1) % batch_size]
            logit[i][easy_mask_list[i]] = easy_logit_list[(i + 1) % batch_size]

        elif mix_type == "easy_hard":
            image[i][easy_mask_list[i]] = hard_image_list[(i + 1) % batch_size]
            label[i][easy_mask_list[i]] = hard_label_list[(i + 1) % batch_size]
            logit[i][easy_mask_list[i]] = hard_logit_list[(i + 1) % batch_size]

        elif mix_type == "easy_rand":
            image[i][rand_mask_list[i]] = image[(i + 1) % batch_size][rand_mask_list[i]]
            label[i][rand_mask_list[i]] = label[(i + 1) % batch_size][rand_mask_list[i]]
            logit[i][rand_mask_list[i]] = logit[(i + 1) % batch_size][rand_mask_list[i]]

    image = image.permute(0, 3, 1, 4, 2, 5)
    image = torch.flatten(image, start_dim=2, end_dim=3)
    new_image = torch.flatten(image, start_dim=3, end_dim=4)

    label = label.permute(0, 1, 3, 2, 4)
    label = torch.flatten(label, start_dim=1, end_dim=2)
    new_label = torch.flatten(label, start_dim=2, end_dim=3)

    logit = logit.permute(0, 1, 3, 2, 4)
    logit = torch.flatten(logit, start_dim=1, end_dim=2)
    new_logit = torch.flatten(logit, start_dim=2, end_dim=3)

    return new_image, new_label, new_logit


def generate_patch_mask(hard_ratio, ex_ratio=40, alpha=30):
    patch_size = hard_ratio.shape[0] * hard_ratio.shape[1]
    idx = round(patch_size * int(ex_ratio) / 100)
    hard_line, indices = torch.sort(hard_ratio.view(-1))

    easy_mask = torch.zeros((patch_size,))
    easy_mask[indices[:idx + 1]] = True
    easy_mask = easy_mask.reshape(hard_ratio.shape[0], hard_ratio.shape[1]).bool()

    hard_mask = torch.zeros((patch_size,))
    hard_mask[indices[-idx - 1:]] = True
    hard_mask = hard_mask.reshape(hard_ratio.shape[0], hard_ratio.shape[1]).bool()

    take_length = round(patch_size * (1 - int(alpha) / 100))
    take_num = round(np.random.uniform(patch_size * 0.1, patch_size * 0.5))
    rand_mask = torch.zeros((patch_size,))
    rand_indices = random.sample(range(0, take_length), take_num)
    rand_mask[indices[rand_indices]] = True
    rand_mask = rand_mask.reshape(hard_ratio.shape[0], hard_ratio.shape[1]).bool()

    return easy_mask, hard_mask, rand_mask


def get_patchs(input, patch_num):
    assert input.shape[-1] % patch_num == 0
    input = input.unsqueeze(1)
    input1 = torch.chunk(input, int(patch_num), dim=-1)
    input1 = torch.cat(input1, dim=1)
    input1 = input1.unsqueeze(1)
    output = torch.chunk(input1, int(patch_num), dim=-2)
    output = torch.cat(output, dim=1)
    return output


def patch_contrast(
        prob_l_teacher,
        prob_u_teacher,
        rep_all,
        rep_all_teacher,
        label_l_small,
        label_u_small,
        patch_num,
        feat_size,
        temp,
):
    assert feat_size[0] % patch_num == 0
    assert feat_size[1] % patch_num == 0

    num_labeled, num_class, _, _ = prob_l_teacher.shape
    label_l = F.interpolate(label_l_small, size=feat_size, mode="nearest")
    label_u = F.interpolate(label_u_small, size=feat_size, mode="nearest")
    prob_l_teacher = F.interpolate(prob_l_teacher, size=feat_size, mode="bilinear", align_corners=True)
    prob_u_teacher = F.interpolate(prob_u_teacher, size=feat_size, mode="bilinear", align_corners=True)
    rep_student = F.interpolate(rep_all, size=feat_size, mode="bilinear", align_corners=True)
    rep_teacher = F.interpolate(rep_all_teacher, size=feat_size, mode="bilinear", align_corners=True)

    label_l_patchs = get_patchs(label_l, patch_num)
    label_u_patchs = get_patchs(label_u, patch_num)
    prob_l_patchs = get_patchs(prob_l_teacher, patch_num)
    prob_u_patchs = get_patchs(prob_u_teacher, patch_num)
    rep_student_patchs = get_patchs(rep_student, patch_num)
    rep_teacher_patchs = get_patchs(rep_teacher, patch_num)

    hard_ratio_l = compute_hard_ratio(prob_l_patchs)
    hard_ratio_u = compute_hard_ratio(prob_u_patchs)

    con_dist_l = torch.ones(num_labeled, patch_num, patch_num)
    dist_l = torch.sum(label_l_patchs, dim=[-2, -1]).cpu().bool()
    for i in range(dist_l.shape[-1]):
        con_dist_l *= dist_l[:, :, :, i]
    con_dist_u = torch.ones(num_labeled, patch_num, patch_num)
    dist_u = torch.sum(label_u_patchs, dim=[-2, -1]).cpu().bool()
    for i in range(dist_u.shape[-1]):
        con_dist_u *= dist_u[:, :, :, i]

    if torch.sum(con_dist_u) == 0:
        return torch.tensor(0.0) * rep_student.sum()

    rep_u_stu = rep_student_patchs[num_labeled:]
    rep_l_tea = rep_teacher_patchs[:num_labeled]
    con_u_feat = rep_u_stu[con_dist_u.bool()]
    con_u_ratio = hard_ratio_u[con_dist_u.bool()]
    con_u_label = label_u_patchs[con_dist_u.bool()]
    con_l_feat = rep_l_tea[con_dist_l.bool()]
    con_l_ratio = hard_ratio_l[con_dist_l.bool()]
    con_l_label = label_l_patchs[con_dist_l.bool()]
    lu_dist = count_lu_couple(con_l_ratio, con_u_ratio)

    con_l_feat = con_l_feat.permute(0, 2, 3, 1)
    con_u_feat = con_u_feat.permute(0, 2, 3, 1)

    proto_list = []
    for c in range(num_class):
        c_mask = con_l_label[:, c]
        temp_dist = []
        for i in range(con_l_label.shape[0]):
            i_mask = c_mask[i]
            i_feat = con_l_feat[i]
            temp_dist.append(torch.mean(i_feat[i_mask.bool()].detach(), dim=0, keepdim=True))
        proto = torch.cat(temp_dist, dim=0)
        proto_list.append(proto.unsqueeze(1))
    prototype = torch.cat(proto_list, dim=1)

    con_mask_list = []
    for i in range(len(con_u_feat)):
        i_feat = con_u_feat[i]
        i_proto = prototype[lu_dist[i]]
        similar_list = []
        for c in range(num_class):
            similar_list.append(torch.matmul(i_feat, i_proto[c]).unsqueeze(-1))
        similar_map = torch.cat(similar_list, dim=-1)
        similar_map = F.normalize(similar_map, p=2, dim=-1)
        con_prob = F.softmax(similar_map, dim=-1)
        _, con_mask = torch.max(con_prob, dim=-1)
        con_mask_list.append(con_mask.unsqueeze(0))
    con_mask = torch.cat(con_mask_list, dim=0)
    con_mask = F.one_hot(con_mask, num_class).float().permute(0, 3, 1, 2)

    inter_mask = con_mask * con_u_label

    loss_contrast_list = []
    for i in range(len(con_u_feat)):
        i_feat = con_u_feat[i]
        i_proto = prototype[lu_dist[i]]
        for c in range(num_class):
            c_mask = inter_mask[i, c]
            anchor_feat = i_feat[c_mask.bool()].clone().cuda()
            num_queries = anchor_feat.shape[0]
            with torch.no_grad():
                positive_feat = i_proto[c].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1).cuda()
                negative_feat = i_proto[1 - c].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1).cuda()
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)
            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=-1)
            loss_contrast = F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda())
            if loss_contrast != loss_contrast:
                continue
            loss_contrast_list.append(loss_contrast.unsqueeze(0))

    if len(loss_contrast_list) == 0:
        return 0 * rep_student.sum()

    loss_contrast = torch.cat(loss_contrast_list)
    loss_contrast = torch.mean(loss_contrast)

    return loss_contrast


def count_lu_couple(ratio_l, ratio_u):
    assert len(ratio_l.shape) == 1
    assert len(ratio_u.shape) == 1
    dist = torch.abs(ratio_u.unsqueeze(-1) - ratio_l.unsqueeze(0))
    min_index = torch.argmin(dist, dim=-1)

    return min_index


if __name__ == "__main__":
    image = torch.randn(4, 3, 513, 513)
    label = torch.ones((4, 513, 513))
    logit = torch.randn(4, 513, 513)
    pred = torch.randn(4, 2, 513, 513)
    patch_num = 9
    ex_ratio = 20
    output = devide_into_patchs(image, label, logit, pred, patch_num)
    hard_ratio = compute_hard_ratio(output[-1])
    new_image, new_label, new_logit = generate_patch_data(output[0], output[1], output[2], hard_ratio, ex_ratio)
    print("Finish Debug")
