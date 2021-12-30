set -x

# bash exp/tests/setup_env.sh

bucket=${1:-bucket-3690}


pip install --no-cache-dir tl2

# torch
python -m tl2.modelarts.scripts.copy_tool \
  -s s3://$bucket/ZhouPeng/pypi/torch182_cu102_py37 -d /cache/pypi -t copytree
for filename in /cache/pypi/*.whl; do
    pip install $filename
done

pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir ninja
pip install -e torch_fidelity_lib

pip uninstall -y tl2




