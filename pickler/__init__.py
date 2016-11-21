import os

pickler_main = os.path.dirname(__file__)
print("Pickler<0>: ", pickler_main)


def pickler(identifier, callable, *args, **kwargs):
    """
   Simplify pickle stuff.


   :param identifier: filename - will be saved in projectdir/pickler/identifier.pickler
   :param callable: function to generate the data if no cached data could be loaded.
   :param args: list of arguments for callable - you can keep the signature of callable. Just look at example
   :param kwargs: dict of arguments for callable - you can keep the signature of callable. Just look at example
   :return:
   :Example:
   >>> def generate_stuff(a, b, c):
   >>>    return "stuff"
   >>> stuff1 = pickler("my_stuff_abc", generate_stuff, "a", "b", "c")
   >>> stuff2 = pickler("my_stuff_dict", generate_stuff, a="a", b="b", c="c")

   """
    import os, pickle
    file_path = os.path.join(pickler_main, identifier + ".pickler")
    try:
        foo = pickle.load(open(file_path, "rb"))
        print("Pickler<{}>: Persisted dataset loaded. ".format(identifier))
    except (OSError, IOError, FileNotFoundError):
        print("Pickler<{}>: Load failured ... generate".format(identifier))
        foo = callable(*args, **kwargs)
        print("Pickler<{}>: Generated ... dump".format(identifier))
        try:
            pickle.dump(foo, open(file_path, "wb"))
            print("Pickler<{}>: Dumped".format(identifier))
        except (OSError, IOError, FileNotFoundError, pickle.PickleError):
            pass
    return foo
