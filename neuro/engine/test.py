class Tensor: 
    def __init__(self, name, value, parent):
        self.name = name  # Add a name attribute
        self.value = value
        self.parent = parent

    def __repr__(self):
        return f"Node {self.name}, value={self.value}"
    def mul(self, node2):
        self.value *=node2.value
        return self.value



print(a.mul(b))