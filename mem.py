import torch
max_indices = torch.tensor([1,2,3,2,4,5])
ii = torch.tensor([1,2,3,2,4,5])
print(ii * max_indices)
i = 2
idx_o = torch.nonzero(max_indices == i)
print(idx_o)


feats = torch.tensor([[[1,2,3,4,5,6],
                      [4,5,6,7,8,9],
                      [2,3,4,5,6,7],
                      [3,4,5,6,7,8],

                      [5,6,7,8,9,10]]])
keys = torch.tensor([[1,3,5,7,9,11],
                     [2,4,6,8,10,12],
                     [3,6,9,12,15,18]])

batch_size, N, dims = feats.size()
feats = feats.contiguous().view(batch_size * N, dims)
sim = torch.matmul(feats, torch.t(keys))
sim_max, _ = torch.max(sim, 1)
print(sim)
print(sim_max)
idx_keep = sim_max >= 400
idx_extra = torch.where(sim_max < 400)[0][:2]
print(idx_extra)
feats_keep = feats[idx_keep]
feats_extra = feats[idx_extra]
print(feats_keep)
print(feats_extra)
