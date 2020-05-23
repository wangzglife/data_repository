# module_9492.pth
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
