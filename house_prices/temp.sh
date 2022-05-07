for f in `find . -name "*.py"`; do autopep8 --in-place --select=000 $f; done
