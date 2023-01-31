

# black
black --line-length 80 ./src
# mypy
mypy --strict --follow-imports skip src > mypy.txt
# flake8
flake8 --statistics --show-source --ignore E203,W503,E231 --max-line-length 80 ./src > flake8.txt
