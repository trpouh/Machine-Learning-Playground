class Hawk:

    layers = {}

    def __init__(self):
        pass

    def layer(self, layer, index=-1):

        if not bool(self.layers):
            self.layers[max(0, index)] = layer
            return self

        key = index if index > -1 else max(*[self.layers])+10
        self.layers[key] = layer

        return self
