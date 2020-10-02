import torch
import torch.autograd.profiler as profiler

a=torch.FloatTensor(1000,256).normal_()
with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    m = {}
    for i in range(a.size(0)-1):
        m[i]=a[i]

    a = torch.stack(list(m.values()),dim=0)

print(a.size())
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
