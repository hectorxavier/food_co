import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px

# Carga de datos
data = pd.read_csv('logs_exp_us.csv', sep= '\t')
# Preparacion de datos
data.columns = ['event_name', 'user_id', 'timestamp', 'group']
data['timestamp'] = pd.to_datetime(data['timestamp'], unit= 's')
data['date'] = data['timestamp'].dt.date
data.info()
print(data.head())
# No se registran valores ausentes.

# Revision de duplicados
print(data[data.duplicated()])
data.drop_duplicates(inplace= True)
## Existen 413 filas duplicadas, se eliminaran las filas duplicadas
print(data[data.duplicated()])
# ¿Cuántos eventos hay en los registros?
print('Una vez eliminados los registros duplicados se mantienen ' + str(len(data['event_name'])) + ' eventos.')
# ¿Cuántos usuarios y usuarias hay en los registros?
print('Número de usuarios:' + str(data['user_id'].nunique()))
# ¿Cuál es el promedio de eventos por usuario?
events_per_user = data.groupby('user_id').agg({'event_name': 'count'})
print('El promedio de eventos por usuario es: '+ str(events_per_user['event_name'].mean().round(0)))

print(data['date'].min(), data['date'].max())
plt.figure(figsize=(20,5))
fig = sns.histplot(data, x = 'date')
fig.tick_params(axis='x', labelrotation = 90)
plt.close()
fig = sns.boxplot(data, x = 'date')
fig.tick_params(axis='x', labelrotation = 45)
plt.close()

