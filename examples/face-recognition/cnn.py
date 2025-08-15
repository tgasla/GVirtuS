 
# nvcc -shared -Xcompiler -fPIC -o libextension.so extension.cu -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcudart -lcudnn -lcublas
import time
import ctypes
import numpy as np
from PIL import Image
import os
import torch 
import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
import torch.nn.functional as F


libextension = ctypes.CDLL("./libextension.so")

test_data = torch.load("test_data.pt")
test_images, test_labels = test_data

class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust based on pooling
        self.fc2 = nn.Linear(128, 40)  # 40 classes (people)

    def forward(self, x):
        x=self.conv1(x)
        x = F.max_pool2d(x, 2)  # Downscale (64 → 32)
        x=self.conv2(x)
        x = F.max_pool2d(x, 2)  # Downscale (32 → 16)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)  

# Load the trained model
model = FaceCNN()  # Ensure model structure is defined
model.load_state_dict(torch.load("face_cnn.pth"))
model.eval()

# Extract weights and biases
weights_conv1 = model.conv1.weight.detach().numpy().astype(np.float32) 
bias_conv1 = model.conv1.bias.detach().numpy().astype(np.float32) 
weights_conv2 = model.conv2.weight.detach().numpy().astype(np.float32) 
bias_conv2 = model.conv2.bias.detach().numpy().astype(np.float32) 
weights_fc1 = model.fc1.weight.detach().numpy().astype(np.float32) 
bias_fc1 = model.fc1.bias.detach().numpy().astype(np.float32)  
weights_fc2 = model.fc2.weight.detach().numpy().astype(np.float32)  
bias_fc2 = model.fc2.bias.detach().numpy().astype(np.float32) 

weights_dict = {
    "weights_conv1": weights_conv1,
    "bias_conv1": bias_conv1,
    "weights_conv2": weights_conv2,
    "bias_conv2": bias_conv2,
    "weights_fc1": weights_fc1,
    "bias_fc1": bias_fc1,
    "weights_fc2": weights_fc2,
    "bias_fc2": bias_fc2
}

# class CNN():
#     def __init__(self):
#         super(CNN, self).__init__()

#         self.weights_dict=weights_dict

#     def forward(self, x, output):
#         x_in=x.reshape(64,64).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         weights1=self.weights_dict["weights_conv1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         bias1=self.weights_dict["bias_conv1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         weights2=self.weights_dict["weights_conv2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         bias2=self.weights_dict["bias_conv2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         weightsfc1=self.weights_dict["weights_fc1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         biasfc1=self.weights_dict["bias_fc1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         weightsfc2=self.weights_dict["weights_fc2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         biasfc2=self.weights_dict["bias_fc2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         #x = conv(x,self.weights, self.bias)
#         #x = subsample(x,self.subsample_weights, self.subsample_bias)
#         #x = fully_connect(x,self.weights_fc,self.bias_fc)
#         libextension.forward_pass(x_in, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), weights1, bias1, weights2, bias2,weightsfc1, biasfc1, weightsfc2, biasfc2)
#         return x

# # Instantiate the model
# model = CNN()

# def test_model():
#     libextension.forward_pass.argtypes = [
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float), 
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float), 
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float),  
#         ctypes.POINTER(ctypes.c_float),  
#         ]
#     # model.eval()  # Set the model to evaluation mode
#     correct = 0
#     total = 0
#     directory="images/"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     total=10
#     for i in range(total):
#         inputs=test_images[i].numpy().squeeze()
#         labels=test_labels[i].numpy().squeeze()
#         image = Image.fromarray((inputs*255).astype(np.uint8))
#         image.save(f"{directory}test_image_{i}.png",)
#         # image.show()

#         outputs=np.zeros((40,), dtype=np.float32)
#         model.forward(inputs,outputs)
#         predicted = np.argmax(outputs)
#         print(predicted,labels)
#         correct += (predicted == labels).sum().item()

#     print(f"Test Accuracy: {100 * correct / total}%")

# Evaluate the model
def test_model():
    weights1=weights_dict["weights_conv1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias1=weights_dict["bias_conv1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    weights2=weights_dict["weights_conv2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias2=weights_dict["bias_conv2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    weightsfc1=weights_dict["weights_fc1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    biasfc1=weights_dict["bias_fc1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    weightsfc2=weights_dict["weights_fc2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    biasfc2=weights_dict["bias_fc2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    
    libextension.create_model.restype = ctypes.c_void_p
    libextension.create_model.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # weights_conv1
        ctypes.POINTER(ctypes.c_float),  # bias_conv1
        ctypes.POINTER(ctypes.c_float),  # weights_conv2
        ctypes.POINTER(ctypes.c_float),  # bias_conv2
        ctypes.POINTER(ctypes.c_float),  # weights_fc1
        ctypes.POINTER(ctypes.c_float),  # bias_fc1
        ctypes.POINTER(ctypes.c_float),  # weights_fc2
        ctypes.POINTER(ctypes.c_float)   # bias_fc2
    ]
    libextension.forward_pass.restype = None
    libextension.forward_pass.argtypes = [
        ctypes.c_void_p,  
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),  
        ]

    Model=libextension.create_model(weights1, bias1, weights2, bias2,weightsfc1, biasfc1, weightsfc2, biasfc2)

    correct = 0
    directory="images/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    total=10
    for i in range(total):
        inputs=test_images[i].numpy().squeeze()
        labels=test_labels[i].numpy().squeeze()
        image = Image.fromarray((inputs*255).astype(np.uint8))
        image.save(f"{directory}test_image_{i}.png",)
        # image.show()

        x_in=inputs.reshape(64,64).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        outputs=np.zeros((40,), dtype=np.float32)
        libextension.forward_pass(Model,x_in, outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        predicted = np.argmax(outputs)
        # print(predicted,labels)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total}%")
    # libextension.delete_model(Model)

if __name__ == "__main__":
    total=0
    for i in range(1):
        start = time.time()
        test_model()
        end = time.time()
        print(f"Execution Time: {end - start:.6f} seconds")
        total=total+end - start
    print(f"Average Time: {total/10:.6f} seconds")