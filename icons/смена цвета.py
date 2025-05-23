import re

with open(r'C:\Users\Artem\Desktop\примеры pyqt\приложение\icons\icons8_separate_document_1.svg', 'r', encoding='utf-8') as file:
    svg_data = file.read()

# Новый цвет
new_color = '#000000'

# Заменим все значения fill="...":
updated_svg = re.sub(r'fill="[^"]+"', f'fill="{new_color}"', svg_data)

with open(r'updated_icons8_separate_document_1.svg', 'w', encoding='utf-8') as file:
    file.write(updated_svg)

print("Цвет всех fill изменён.")