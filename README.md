# corticalMeasurement

archivo main => para un unico hueso.

archivo mainAll => para todos los huesos, cogiendo como referencia el hueso especificado en file.ini (reference_bone).

archivo mainP => archivo para realizar pruebas de las correcciones en alineamiento y orientación y obtención de espesores (para no tener que ejecutar el mainAll entero). Hay que hardcodear las variables:

* ruta y nombre del STL de referencia
	
		path = "C:/Users/lfsanchez/OneDrive/COMILLAS/Workspace/dicom/data/stl/" # linea 160
		reader_r.SetFileName(path+"leg7.stl") # linea 167

* ruta del STL a orientar y alinear
	
		reader_c.SetFileName(path+"femur1Der.stl") # linea 406

## Dependences:
See requirements.txt for tested library versions

pip install visual-logging
pip install vg
