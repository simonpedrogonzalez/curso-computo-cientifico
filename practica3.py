import warnings


class SilencedWarning(UserWarning):
    ...


def silence(*args):
    errors = tuple(args)

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as err:
                if isinstance(err, errors):
                    warnings.warn(str(err), SilencedWarning)
                else:
                    raise

        return wrapper

    return decorator


@silence(ZeroDivisionError, TypeError)
def foo(a, b):
    return a / b


foo(1, "coso")
