from __future__ import print_function
import builtins as __builtin__
import inspect

def print(content):
    """Print content with caller information in brackets"""
    stack = inspect.stack()
    caller_instance = stack[1][0]
    # Get caller attributes
    class_name = None
    if "self" in caller_instance.f_locals.keys():
        class_name = str(caller_instance.f_locals["self"].__class__.__name__)
    func_name = str(caller_instance.f_code.co_name)
    file_name = str(caller_instance.f_code.co_filename)
    # Construct print name
    caller_name = file_name
    if class_name:
        caller_name = class_name+"."+func_name
    elif func_name != "<module>":
        caller_name = func_name
    __builtin__.print("["+caller_name+"] "+str(content))
    return
