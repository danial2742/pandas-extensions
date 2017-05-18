# What?

A set of extension functions for Pandas DataFrame.

# Why?

I find parts of the Pandas syntax difficult weird and difficult to learn and read. This library contains extension functions with friendly names that can be used in two ways:

* as a usual function, passing `DataFrame` as the first parameter;
* monkey-patched case, using `pext.patch.monkey_patch_data_frame`. This way all the functions will be added directly to the `DataFrame` class.

