import matplotlib.pyplot as plt
import main

# Пример списков координат (замените на свои данные)
x = main.common_wave
y = main.read_oneFile("control/mk1/prob.txt", True)

# Построение точечного графика
plt.scatter(x, y, color='blue', label='Точки')

# Добавление заголовка и подписей осей
plt.title('Визуализация точек')
plt.xlabel('Координата X')
plt.ylabel('Координата Y')

# Отображение легенды
plt.legend()

# Показать сетку для удобства
plt.grid(True)

# Вывод графика на экран
plt.show()
