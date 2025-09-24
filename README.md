Cómo ejecutar:
1) source .venv/bin/activate
2) pip install -r requirements.txt
3) Poner la imagen en data/imagen_profesor.tif (o .png/.jpg/.npy)
4) Ajustar GREEN_IDX y NIR_IDX en run_ndwi.py
5) python run_ndwi.py  -> guarda outputs/*.png
6) pytest -q          -> ejecuta tests
