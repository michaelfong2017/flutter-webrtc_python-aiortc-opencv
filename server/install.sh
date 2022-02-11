env_name="linux-env"

# For Mac M1, use miniconda3 and then python virtual environment.

python3 -m venv $env_name

$env_name/bin/python -m pip install --upgrade pip
$env_name/bin/pip install wheel
$env_name/bin/pip install black
$env_name/bin/pip install ipykernel
$env_name/bin/pip install -r requirements.txt