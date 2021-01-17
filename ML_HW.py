#!/usr/bin/env python
# coding: utf-8
# Кильдеев Заур, 4бПМ
# # Машинное обучение
# 
# Практическое задание 1 посвящено изучению основных библиотек для анализа данных, а также линейных моделей и методов их обучения. Вы научитесь:
#  * применять библиотеки NumPy и Pandas для осуществления желаемых преобразований;
#  * подготавливать данные для обучения линейных моделей;
#  * обучать линейную, Lasso и Ridge-регрессии при помощи модуля scikit-learn;
#  * реализовывать обычный и стохастический градиентные спуски;
#  * обучать линейную регрессию для произвольного функционала качества.

# ## Библиотеки для анализа данных
# 
# ### NumPy
# 
# Во всех заданиях данного раздела запрещено использовать циклы  и list comprehensions. Под вектором и матрицей в данных заданиях понимается одномерный и двумерный numpy.array соответственно.

# In[22]:


import numpy as np


# Реализуйте функцию, возвращающую максимальный элемент в векторе x среди элементов, перед которыми стоит нулевой. Для x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0]) ответом является 5. Если нулевых элементов нет, функция должна возвращать None.
# 

# In[23]:


def max_element(arr):
    arr = np.array([6, 2, 0, 3, 0, 0, 5, 8,])
    x = arr == 0
print ([1:][arr[:-1]].max())
x = np.array([0, 1, 0, 3, 0, 0, 5, 7, 0])
print(np.max(np.take(x, np.where(x[1:] == 0))))


# In[25]:


x = np.array([0, 1, 0, 3, 0, 0, 5, 7, 0])
print(np.max(np.take(x, np.where(x[:1] == 0))))


# Реализуйте функцию, принимающую на вход матрицу и некоторое число и возвращающую ближайший к числу элемент матрицы. Например: для X = np.arange(0,10).reshape((2, 5)) и v = 3.6 ответом будет 4.

# In[7]:


def nearest_value(X, v):
    X = X.ravel()
    print(X)
    idx = np.abs(X - v).argmin()
    return X[idx]

Z = np.arange(0,10).reshape((2, 5))
Z[0, 0] = 3
v = 2.2
print(Z)
print('Nearest value to {} is {}'.format(v, nearest_value(Z, v)))


# Реализуйте функцию scale(X), которая принимает на вход матрицу и масштабирует каждый ее столбец (вычитает выборочное среднее и делит на стандартное отклонение). Убедитесь, что в функции не будет происходить деления на ноль. Протестируйте на случайной матрице (для её генерации можно использовать, например, функцию [numpy.random.randint](http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html)).

# In[25]:


def scale(X):
    eps = 0.0000001
    e = np.mean(X, axis=0)
    d = np.std(X, axis=0)
    is_not_zero = np.abs(d) > eps
    X[:, is_not_zero] = (X - e)[:, is_not_zero] / d[is_not_zero]
    return X
        
X = scale(np.array([[1, 2, 0], [1, 5, 0], [2, 8, 0], [2, 3, 1]], dtype=float))
print (X)
X = scale(np.array([[1, 2, 2], [1, 5, 2], [2, 8, 2], [2, 3, 2]], dtype=float))
print (X)


#  Реализуйте функцию, которая для заданной матрицы находит:
#  - определитель
#  - след
#  - наименьший и наибольший элементы
#  - норму Фробениуса
#  - собственные числа
#  - обратную матрицу
# 
# Для тестирования сгенерируйте матрицу с элементами из нормального распределения $\mathcal{N}$(10,1)

# In[9]:


from numpy import linalg as la
def get_stats(matrix):
    matrix = np.random.normal(10, 1, (10, 10))
print(la.det(matrix))
print(np.trace(matrix))
print(np.min(matrix))
print(np.max(matrix))
print(la.norm(matrix, ord=2))
print(la.norm(matrix, ord='fro'))
print(la.eig(matrix)[0])
print(la.inv(matrix))


# Повторите 100 раз следующий эксперимент: сгенерируйте две матрицы размера 10×10 из стандартного нормального распределения, перемножьте их (как матрицы) и найдите максимальный элемент. Какое среднее значение по экспериментам у максимальных элементов? 95-процентная квантиль?

# In[22]:


values = np.empty(100)
for exp_num in range(100):
    m1 = np.random.normal(0, 1, (10, 10))
    m2 = np.random.normal(0, 1, (10, 10))
    values[exp_num] = np.max(m1.dot(m2))
print(np.average(values))
print(np.percentile(values, 95))


# ### Pandas
# 
# ![](https://metrouk2.files.wordpress.com/2015/10/panda.jpg)
# 
# #### Ответьте на вопросы о данных по авиарейсам в США за январь-апрель 2008 года.
# 
# [Данные](https://www.dropbox.com/s/dvfitn93obn0rql/2008.csv?dl=0) и их [описание](http://stat-computing.org/dataexpo/2009/the-data.html)

# In[5]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv(r"C:\Users\Заур\Downloads\2008.csv", error_bad_lines=False)
data.head()


# Какая из причин отмены рейса (`CancellationCode`) была самой частой? (расшифровки кодов можно найти в описании данных)

# In[7]:


codes = data["CancellationCode"]
codes.value_counts() #A = carrier


#  Найдите среднее, минимальное и максимальное расстояние, пройденное самолетом.

# In[12]:


dists = data["Distance"]
print ("Max:", dists.max())
print ("Average: ", dists.mean())
print ("Min: ", dists.min())


#  Не выглядит ли подозрительным минимальное пройденное расстояние? В какие дни и на каких рейсах оно было? Какое расстояние было пройдено этими же рейсами в другие дни?

# In[26]:


date_fnum_ucarrier = ["Year", "Month", "DayofMonth", "FlightNum", "UniqueCarrier"]
fnum_ucarrier_dist = ["FlightNum", "UniqueCarrier", "Distance"]
days = data[dists == dists.min()][date_fnum_ucarrier].drop_duplicates()

print (days)

flights = data[dists == dists.min()][["FlightNum", "UniqueCarrier"]].drop_duplicates()
get_other_flights = lambda flight:     np.logical_and(data["FlightNum"] == flight[0], data["UniqueCarrier"] == flight[1])

[data[get_other_flights(flight)][fnum_ucarrier_dist].drop_duplicates() for flight in flights.values]


# Из какого аэропорта было произведено больше всего вылетов? В каком городе он находится?

# In[16]:


airports = pd.read_csv("http://stat-computing.org/dataexpo/2009/airports.csv")
most_frequent_airport = data["Origin"].value_counts().index[0]
airports[airports["iata"] == most_frequent_airport]


# Найдите для каждого аэропорта среднее время полета (`AirTime`) по всем вылетевшим из него рейсам. Какой аэропорт имеет наибольшее значение этого показателя?

# In[18]:


origin_mean_airtime = data.groupby("Origin")["AirTime"].aggregate(np.mean)
print (origin_mean_airtime.max())
airport_max = origin_mean_airtime.idxmax()
airports[airports["iata"] == airport_max]


# Найдите аэропорт, у которого наибольшая доля задержанных (`DepDelay > 0`) рейсов. Исключите при этом из рассмотрения аэропорты, из которых было отправлено меньше 1000 рейсов (используйте функцию `filter` после `groupby`).

# In[20]:


threshold = 1000
all_flights = data.groupby("Origin").size()
delayed_flights = data[data["DepDelay"] > 0].groupby("Origin").size()
fraction_delayed = delayed_flights[all_flights > threshold] / all_flights[all_flights > threshold]
max_fraction_delayed = fraction_delayed.idxmax()
print (fraction_delayed.max())
airports[airports["iata"] == max_fraction_delayed]
