class UndefinedFunctionError(Exception):
    def __init__(self, function_name):
        self.function_name = function_name
        super().__init__(f"Undefined file format: {function_name}")
