

# black
black --line-length 80 ./src
# mypy
mypy --ignore-missing-imports --disallow-untyped-calls --disallow-untyped-defs --strict --warn-return-any --disallow-subclassing-any --html-report ./mypyreport ./src > mypy.txt
# flake8
flake8 --statistics --show-source --ignore E203,W503,E231 --max-line-length 80 ./src > flake8.txt
