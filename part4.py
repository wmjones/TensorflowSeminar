# Part 4 Introduction to Classes and super()


class SimpleClass():
    def __init__(self):
        print("hello from simple class")

    def foo(self):
        print("hello from foo")


x = SimpleClass()
x.foo()
print(type(x))


class ExtendedSimpleClass(SimpleClass):
    def __init__(self):
        super().__init__()
        print("hello from extended class")


y = ExtendedSimpleClass()
y.foo()
