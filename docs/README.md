Building Documentation
----------------------
Sphinx and sphinx-rtd-theme are required to build documentation. After this,
simply run the make command for html (or other output type). The resulting files
will be placed into stj_pv/docs/_build/html

    pip install sphinx sphinx-rtd-theme numpydoc
    cd stj_pv/docs
    make html


Resources about documentation
-----------------------------
[Generating documentation](https://developer.ridgerun.com/wiki/index.php/How_to_generate_sphinx_documentation_for_python_code_running_in_an_embedded_system)


NumpyDoc Style examples
-----------------------
Code in this project is documented in the style of numpydoc, and uses this sphinx extension to automatically generate the formatted html

[Numpydoc tutorial](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)


