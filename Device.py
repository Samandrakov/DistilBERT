import torch

# # Проверка доступности CUDA
# if torch.cuda.is_available():
#     # Получение информации о доступных устройствах CUDA
#     device_count = torch.cuda.device_count()
#     print(f"Доступно устройств CUDA: {device_count}")
#     for i in range(device_count):
#         device_name = torch.cuda.get_device_name(i)
#         print(f"Устройство {i}: {device_name}")
# else:
#     print("CUDA не доступна на данной системе.")


x = torch.tensor([1.0, 2.0, 3.0])
x = x.to('cuda')
print(x)