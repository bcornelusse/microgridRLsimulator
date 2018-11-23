
#!/bin/bash

python3 -m venv mgenv
source mgenv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

deactivate
