import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import datetime as dt
from plotly import graph_objects as go
from scipy import stats as st
import math as mth

# Carga de datos
data = pd.read_csv('logs_exp_us.csv', sep= '\t')
# Preparacion de datos
data.columns = ['event_name', 'user_id', 'timestamp', 'group']
data['user_id'] = data['user_id'].astype(str)
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

# Excluir fechas con pocos datos

data_filtered = data[data['date'] > dt.date(2019,7,31)]
fig = sns.histplot(data_filtered, x = 'date')
fig.tick_params(axis='x', labelrotation = 90)
plt.close()

# ¿Perdiste muchos eventos y usuarios al excluir los datos más antiguos?
print('Una vez eliminados los registros con fechas antiguas se mantienen ' + str(len(data_filtered['event_name'])) + ' eventos.')
# Se eliminaron alrededor de 3000 registros, valor cercano al 2% de los registros, no es significativo.

# Asegúrate de tener usuarios y usuarias de los tres grupos experimentales.
print(data_filtered.groupby('group').agg({'user_id' : 'count'}))

#Paso 4. Estudiar el embudo de eventos
#Observa qué eventos hay en los registros y su frecuencia de suceso. Ordénalos por frecuencia.

events_frequency = data_filtered.groupby('event_name').agg({'user_id': 'count'}).sort_values('user_id', ascending = False).reset_index()
events_frequency.columns = ['event_name', 'frequency']
print(events_frequency)

# Encuentra la cantidad de usuarios y usuarias que realizaron cada una de estas acciones. Ordena los eventos por el número de usuarios y usuarias. Calcula la proporción de usuarios y usuarias que realizaron la acción al menos una vez.
events_frequency_by_user = data_filtered.groupby('event_name').agg({'user_id': 'nunique'}).sort_values('user_id', ascending = False).reset_index()
events_frequency_by_user.columns = ['event_name', 'frequency']
print(events_frequency_by_user)

# ¿En qué orden crees que ocurrieron las acciones? ¿Todas son parte de una sola secuencia? No es necesario tenerlas en cuenta al calcular el embudo.
## MainScreenAppear -> OffersScreenAppear -> CartScreenAppear -> PaymentScreenSuccessful // El evento de tutorial influira en el flujo del proceso.


# Utiliza el embudo de eventos para encontrar la proporción de usuarios y usuarias que pasan de una etapa a la siguiente. (Por ejemplo, para la secuencia de eventos A → B → C, calcula la proporción de usuarios en la etapa B a la cantidad de usuarios en la etapa A y la proporción de usuarios en la etapa C a la cantidad en la etapa B).

event_data = pd.pivot_table(data_filtered, values='user_id', columns= 'event_name', aggfunc= lambda x: len(x.unique())).reset_index(drop= True)
print(event_data)

event_data['b_proportion'] = (event_data['OffersScreenAppear'] / event_data['MainScreenAppear'])*100
event_data['c_proportion'] = (event_data['CartScreenAppear'] / event_data['OffersScreenAppear'])*100
event_data['d_proportion'] = (event_data['PaymentScreenSuccessful'] / event_data['CartScreenAppear'])*100
display(event_data)

fig = go.Figure(go.Funnel(x = events_frequency_by_user['frequency'], y = events_frequency_by_user['event_name']))
fig.show()

# ¿En qué etapa pierdes más usuarios y usuarias?
# Con el uso de un embudo en ploty, se observa que la mayor perdida se genera la pasar al evento 'OfferScreenAppear'

# ¿Qué porcentaje de usuarios y usuarias hace todo el viaje desde su primer evento hasta el pago?
event_data['total_process_proportion'] = (event_data['PaymentScreenSuccessful'] / event_data['MainScreenAppear'])*100
print(event_data.iloc[:,-1:].round(2))
# 'Del total de usuario, unicamente llega realizan todo el proceso el 47.7% de los usuarios.'

# Paso 5. Estudiar los resultados del experimento

# ¿Cuántos usuarios y usuarias hay en cada grupo?
group_data = pd.pivot_table(data_filtered, values='user_id', columns= 'group', aggfunc= lambda x: len(x.unique())).reset_index(drop= True)
group_data.columns = ['group_246', 'group_247', 'group_248']
print(group_data)

# Tenemos dos grupos de control en el test A/A, donde comprobamos nuestros mecanismos y cálculos. Observa si hay una diferencia estadísticamente significativa entre las muestras 246 y 247.


