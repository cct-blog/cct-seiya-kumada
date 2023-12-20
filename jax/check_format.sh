# black
black --line-length 128 .


# mypy
#mypy --strict --follow-imports skip regression_with_jax.py > mypy.txt

# flake8
flake8 --statistics --show-source --ignore E203,W503,E231 --max-line-length 128 . > flake8.txt
