# module_9492.pth
B3
private score:0.9038 
public  score:0.9531
## code
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__();

        self.eNet = EfficientNet.from_name('efficientnet-b3')
        self.eNet._fc = nn.Linear(1536,168+11+7)

    def forward(self,input):
        return self.eNet(input)
```
# module_9502.pth
B4 
private score:0.9114
public  score:0.9589
## code
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__();

        self.eNet = EfficientNet.from_name('efficientnet-b4')
        self.eNet._fc = nn.Linear(self.eNet._fc.in_features,168+11+7)

    def forward(self,input):
        return self.eNet(input)
```
# module_9523.pth
B4 
private score:0.9234
public  score:0.9602

soft_label   gt,vd,cd = 0.6,0.8,0.8
## code
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__();

        self.eNet = EfficientNet.from_name('efficientnet-b4')
        self.eNet._fc = nn.Linear(self.eNet._fc.in_features,168+11+7)

    def forward(self,input):
        return self.eNet(input)
```
