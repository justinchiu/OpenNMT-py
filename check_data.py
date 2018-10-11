import torch
import onmt

print(len(torch.load("data/iwslt/iwslt_125.train.0.pt")))
print(len(torch.load("data/iwslt/iwslt_125.valid.0.pt")))
print(len(torch.load("data/iwslt/iwslt_125_test.train.0.pt")))
print(len(torch.load("data/iwslt/iwslt_125_test.valid.0.pt")))
