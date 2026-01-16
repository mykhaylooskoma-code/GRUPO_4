ConflictDamage (Demo CDIO-1)
1) Requisitos

- Python 3.x
- Librerías:
  - numpy
  - pandas
  - pillow


2) Estructura de archivos 
Coloca en la misma carpeta:
- run_demo_conflictdamage.py
- abans.JPG        (imagen PRE / antes)
- despres.JPG      (imagen POST / después)


3) Cómo ejecutar la demo
Ejecutar con imágenes reales (PRE/POST):
  py run_demo_conflictdamage.py --pre abans.JPG --post despres.JPG --tile 64 --thr 0.22

Parámetros importantes:
- --tile : tamaño de cada tesela (cuadrícula) en píxeles. Ej: 64 (más “barrio”), 32 (más detalle).
- --thr  : umbral de cambio para considerar un píxel como “cambio fuerte”. Ej: 0.22 - 0.30.

Ejemplos de ajustes:
- Si detecta demasiado (muchas teselas “dañadas”):
  py run_demo_conflictdamage.py --pre abans.JPG --post despres.JPG --tile 64 --thr 0.30

- Si detecta muy poco:
  py run_demo_conflictdamage.py --pre abans.JPG --post despres.JPG --tile 64 --thr 0.18


4) Qué hace el algoritmo (resumen)

1) Carga la imagen PRE y POST y las pone al mismo tamaño.
2) Convierte ambas a escala de grises [0,1].
3) Calcula un mapa de cambios: change_map = abs(POST - PRE).
4) Divide la imagen en teselas (tile x tile).
5) Para cada tesela calcula damage_score = % de píxeles con cambio > thr.
6) Ordena las teselas por score y marca las más altas como zonas prioritarias.


5) Outputs generados (carpeta outputs/)

Documentos:
- tiles.csv       : tabla con todas las teselas y su damage_score
- tiles.geojson   : polígonos de teselas (en coordenadas de píxel, para demo)
- report.json     : parámetros usados + indicadores + top10 teselas

Imágenes (para visualizar el cálculo):
- pre_gray.png / post_gray.png      : PRE y POST en escala de grises
- change_map.png                    : mapa de cambios normalizado
- change_mask.png                   : máscara binaria (cm > thr)
- pre_grid.png / post_grid.png      : PRE/POST con cuadrícula de teselas
- post_top_tiles.png                : POST con las teselas top marcadas
- tile_score_map.png                : imagen de score por teselas (0..1)

