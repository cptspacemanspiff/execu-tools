import torch



class BaseWrapper(torch.nn.Module):
    # model that is being wrapped:
    model : torch.nn.Module
    
    # cache for model:
    cache : torch.nn.Module

    # list of fqns of shared buffers:
    shared_buffer_fqns : list[str]

    def __init__(self,model,cache):
        super().__init__()
        self.model = model
        self.cache = cache

    def get_shared_buffer_fqns(self):
        return self.shared_buffer_fqns