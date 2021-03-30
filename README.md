# Style transfer
Перенос стиля с одних изображений на другие.
Особенности: в качестве базы и стилей можно использовать несколько изображений, есть дополнительная борьба с шумами на изображении, в качестве начального изображения можно взять как одно из базовых, так и шум.

# Запуск:
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
deactivate
rm -rf venv
