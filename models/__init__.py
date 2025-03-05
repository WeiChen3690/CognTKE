import importlib



def load_model(model_name):
    """
    动态加载模型名
    """
    module_name = f'models.{model_name}'
    module = importlib.import_module(module_name)
    model_class = getattr(module, model_name)
    return model_class