class LazyProperty:
    def __init__(self, func):
        self.func = func
        self.attr_name = f"_{func.__name__}"

    def __get__(self, instance, owner):
        if instance is None:
            return self

        attr_value = getattr(instance, self.attr_name)
        if attr_value is None:
            attr_value = self.func(instance)
            setattr(instance, self.attr_name, attr_value)
        assert attr_value is not None, "`{self.attr_name}` has not been initialized despite the LazyProperty."

        return attr_value
