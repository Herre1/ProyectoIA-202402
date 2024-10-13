#-----------------------------------------------#
#
#   El siguiete codigo une todas las bases de   #
#   datos tomadas de los videos, esto con el    #
#   fin de tener una sola base de datos para    #
#   el posterior analisis                       #
#                                               #
#   SOLO EJECUTAR LUEGO DE TENER TODAS LAS      #
#   BASES DE DATOS CREADAS                      #
#                                               #
#-----------------------------------------------#

import pandas as pd
import os

# Directorio donde están ubicados los archivos CSV
csv_directory = 'Entrega-1/VideosDataSet/'
output_file = 'DataBase/combined_pose_data.csv'

# Lista para almacenar los DataFrames
dataframes = []

# Iteramos sobre todos los archivos en el directorio
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_directory, filename)
        
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenamos todos los DataFrames en uno solo
combined_df = pd.concat(dataframes, ignore_index=True)

# Guardamos el DataFrame combinado en un nuevo archivo CSV
combined_df.to_csv(output_file, index=False)

print(f"Archivos combinados y guardados en {output_file}")