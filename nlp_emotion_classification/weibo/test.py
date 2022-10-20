import torch
import torch.nn as nn

available = torch.cuda.is_available()
print(available)
properties = torch.cuda.get_device_properties(0)
print(properties)

print(torch.cuda.cudaStatus)
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
print(torch.cuda.current_device())

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(torch.device(device))
