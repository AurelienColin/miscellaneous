class LazyProperty:
    """
    The LazyProperty is used to be less verbose. Eg., the following code:
    It implements lazy computation (i.e. computing variables only when needed)
    and run a sanity check (non-None) to ensure when do get stuck in a loop.


    @dataclass
    class Foo:
        _bar: typing.Optional[int] = None

        def bar(self) -> int:
            if self._bar is None:
                self._bar = ...
            assert self._bar is not None
            return self._bar

    is replaced by:

    @dataclass
    class Foo:
        _bar: typing.Optional[int] = None

        @LazyProperty
        def bar(self) -> int:
            return ....
    """

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
