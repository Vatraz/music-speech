def hello_decorator(func):
    def inner1(*args, **kwargs):
        print("before Execution")
        print(args)
        print(kwargs)
        print(func)
        # getting the returned value
        returned_value = func(*args, **kwargs)
        print("after Execution")

        # returning the value to the original frame
        return returned_value

    return inner1


# adding decorator to the function
@hello_decorator
def sum_two_numbers(a, b):
    print("Inside the function")
    return a + b


a, b = 1, 2


def sum21(a, b):
    print("Inside the function")
    return a + b

sum22 = hello_decorator(sum21)

# getting the value through return of the function
print("Sum =", sum_two_numbers(a, b))
print("Sum =", sum22(a, b))