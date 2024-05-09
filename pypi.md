# tar.gz
python setup.py sdist build
# wheels
python setup.py bdist_wheel --universal
# tar.gz and wheels
python3 setup.py sdist bdist_wheel
# upload to pypi
twine upload dist/*