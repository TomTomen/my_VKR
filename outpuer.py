import sys
import json
import time
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout,
    QFrame, QFileDialog, QLabel, QComboBox, QMenu, QSizePolicy,
    QTableWidget, QTableWidgetItem, QStackedWidget, QSpinBox, QDoubleSpinBox,
    QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QProgressDialog
)
# Импорт для моделей ML/DL 
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.callbacks import Callback

from prophet import Prophet as FbProphet 
import logging
import xgboost as xgb 

# Подавляет информационные сообщения от Prophet и CmdStanPy
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from PyQt5.QtCore import Qt, QUrl, QPropertyAnimation, QEasingCurve, QTimer,QSize
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QIcon
import plotly.graph_objects as go
import tempfile
import traceback 

# 
def load_styles_from_json(file_path):
    """Загружает стили из файла JSON и преобразует их в строку таблицы стилей Qt."""

    with open(file_path, 'r', encoding='utf-8') as f:
        styles = json.load(f)

    style_str = ""
    for selector, properties in styles.items(): 
        style_str += f"{selector} {{"
        for prop, value in properties.items():
            style_str += f"{prop}: {value};"
        style_str += "}\n"
    return style_str


# 
class DragDropWidget(QLabel):
    """Виджет QLabel, который поддерживает перетаскивание CSV-файлов."""

    def __init__(self, on_file_dropped):
        super().__init__('Перетащи CSV файл сюда') 
        self.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        self.setStyleSheet("""
            QLabel {
                background-color: #E8F5E9; /* Светло-серый фон */
                color: #333333; /* Темный текст */
                font-size: 18px;
                border: 2px dashed #aaaaaa; /* Серая пунктирная граница */
                border-radius: 20px;
                /* padding: 20px; */ /* Оставляем без изменений */
            }
        """)
        self.setAcceptDrops(True)
        self.on_file_dropped = on_file_dropped 

    def dragEnterEvent(self, event):
        """Принимает событие, если оно содержит URL-адреса включая пути к папкам."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Обрабатывает удаленный файл, вызывая обратный вызов, если это CSV."""
        for url in event.mimeData().urls(): 
            file_path = url.toLocalFile()
            if file_path.endswith('.csv'):
                self.on_file_dropped(file_path)


# 
class LSTMConfigDialog(QDialog):
    """Диалоговое окно для настройки параметров модели LSTM."""
    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Настройка LSTM") 
        layout = QFormLayout(self)

        self.sequence_len_spin = QSpinBox()
        self.sequence_len_spin.setRange(1, 100) 
        self.sequence_len_spin.setValue(5) 
        self.sequence_len_spin.setToolTip("Сколько предыдущих шагов использовать для предсказания следующего") 
        layout.addRow("Длина входной последовательности:", self.sequence_len_spin) 

        self.lstm_units_spin = QSpinBox()
        self.lstm_units_spin.setRange(10, 200) 
        self.lstm_units_spin.setSingleStep(10)
        self.lstm_units_spin.setValue(50) 
        self.lstm_units_spin.setToolTip("Количество нейронов в скрытом слое LSTM") 
        layout.addRow("Количество LSTM юнитов:", self.lstm_units_spin) 

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500) 
        self.epochs_spin.setValue(50) 
        self.epochs_spin.setToolTip("Сколько раз модель просмотрит весь обучающий набор данных") #
        layout.addRow("Количество эпох:", self.epochs_spin) 

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128) 
        self.batch_size_spin.setValue(32) 
        self.batch_size_spin.setToolTip("Количество образцов данных, обрабатываемых за одну итерацию обучения") 
        layout.addRow("Размер батча:", self.batch_size_spin)

        
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addRow(self.buttonBox)


    def get_params(self):
        """Возвращает выбранные параметры в виде справочника."""
        return {
            'sequence_len': self.sequence_len_spin.value(),
            'lstm_units': self.lstm_units_spin.value(),
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value() 
        }

# 
class XGBoostConfigDialog(QDialog):
    """Диалоговое окно для настройки параметров модели XGBoost."""
    def __init__(self, parent=None):
        
        super().__init__(parent)
        self.setWindowTitle("Настройка XGBoost") 
        layout = QFormLayout(self)

        
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 1000)
        self.n_estimators_spin.setSingleStep(10)
        self.n_estimators_spin.setValue(100)
        self.n_estimators_spin.setToolTip("Количество деревьев в ансамбле.") 
        layout.addRow("Количество деревьев (n_estimators):", self.n_estimators_spin)

        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 20)
        self.max_depth_spin.setValue(3)
        self.max_depth_spin.setToolTip("Максимальная глубина каждого дерева.") 
        layout.addRow("Макс. глубина дерева (max_depth):", self.max_depth_spin) 

        self.learning_rate_spin = QDoubleSpinBox() 
        self.learning_rate_spin.setRange(0.01, 1.0)
        self.learning_rate_spin.setSingleStep(0.01)
        self.learning_rate_spin.setDecimals(2)
        self.learning_rate_spin.setValue(0.1)
        self.learning_rate_spin.setToolTip("Скорость обучения (шаг градиентного спуска).") 
        layout.addRow("Скорость обучения (learning_rate):", self.learning_rate_spin) 

        self.subsample_spin = QDoubleSpinBox()
        self.subsample_spin.setRange(0.1, 1.0)
        self.subsample_spin.setSingleStep(0.1) 
        self.subsample_spin.setDecimals(1)
        self.subsample_spin.setValue(0.8)
        self.subsample_spin.setToolTip("Доля обучающих образцов для каждого дерева.") 
        layout.addRow("Доля подвыборки (subsample):", self.subsample_spin) 

        self.colsample_bytree_spin = QDoubleSpinBox()
        self.colsample_bytree_spin.setRange(0.1, 1.0)
        self.colsample_bytree_spin.setSingleStep(0.1)
        self.colsample_bytree_spin.setDecimals(1)
        self.colsample_bytree_spin.setValue(0.8)
        self.colsample_bytree_spin.setToolTip("Доля признаков (столбцов) для каждого дерева.") 
        layout.addRow("Доля признаков для дерева (colsample_bytree):", self.colsample_bytree_spin) 

       
        self.n_lags_spin = QSpinBox()
        self.n_lags_spin.setRange(1, 50)
        self.n_lags_spin.setValue(7)
        self.n_lags_spin.setToolTip("Количество предыдущих значений (лагов) для использования в качестве признаков.") 
        layout.addRow("Количество лагов:", self.n_lags_spin) 

        
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel) 
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addRow(self.buttonBox)


    def get_params(self):
        """Возвращает выбранные параметры в виде словаря."""
        return {
            'n_estimators': self.n_estimators_spin.value(),
            'max_depth': self.max_depth_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'subsample': self.subsample_spin.value(), 
            'colsample_bytree': self.colsample_bytree_spin.value(),
            'n_lags': self.n_lags_spin.value()
        }


class SARIMAConfigDialog(QDialog):
    """Диалоговое окно для настройки параметров модели SARIMA."""
    def __init__(self, current_m=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройка SARIMA") 
        layout = QFormLayout(self)

        
        p_label = QLabel("Несезонный порядок (p, d, q):") 
        p_label.setToolTip("Авторегрессионный (p), Интегрированный (d), Скользящего среднего (q)") 
        layout.addRow(p_label)
        order_layout = QHBoxLayout()
        self.p_spin = QSpinBox()
        self.p_spin.setRange(0, 10); self.p_spin.setValue(1)
        self.d_spin = QSpinBox()
        self.d_spin.setRange(0, 5); self.d_spin.setValue(1)
        self.q_spin = QSpinBox()
        self.q_spin.setRange(0, 10); self.q_spin.setValue(1)
        order_layout.addWidget(QLabel(" p:"))
        order_layout.addWidget(self.p_spin)
        order_layout.addWidget(QLabel(" d:"))
        order_layout.addWidget(self.d_spin)
        order_layout.addWidget(QLabel(" q:"))
        order_layout.addWidget(self.q_spin)
        layout.addRow(order_layout)

        
        P_label = QLabel("Сезонный порядок (P, D, Q, m):") 
        P_label.setToolTip("Сезонные: Авторегрессионный (P), Интегрированный (D), Скользящего среднего (Q), Период (m)") 
        layout.addRow(P_label)
        seasonal_layout = QHBoxLayout()
        self.P_spin = QSpinBox()
        self.P_spin.setRange(0, 10); self.P_spin.setValue(1)
        self.D_spin = QSpinBox()
        self.D_spin.setRange(0, 5); self.D_spin.setValue(1)
        self.Q_spin = QSpinBox()
        self.Q_spin.setRange(0, 10); self.Q_spin.setValue(1)
        self.m_spin = QSpinBox()
        self.m_spin.setRange(0, 366); self.m_spin.setValue(max(0, current_m)) 
        seasonal_layout.addWidget(QLabel(" P:"))
        seasonal_layout.addWidget(self.P_spin)
        seasonal_layout.addWidget(QLabel(" D:"))
        seasonal_layout.addWidget(self.D_spin)
        seasonal_layout.addWidget(QLabel(" Q:"))
        seasonal_layout.addWidget(self.Q_spin)
        seasonal_layout.addWidget(QLabel(" m:"))
        seasonal_layout.addWidget(self.m_spin)
        layout.addRow(seasonal_layout)

        
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addRow(self.buttonBox)

    def get_params(self):
        """Возвращает выбранные параметры в виде справочника."""
        order = (self.p_spin.value(), self.d_spin.value(), self.q_spin.value())
        
        m_val = self.m_spin.value()
        P_val = self.P_spin.value()
        D_val = self.D_spin.value()
        Q_val = self.Q_spin.value()
        if P_val == 0 and D_val == 0 and Q_val == 0:
            m_val = 0 
        elif m_val == 0:
             print("Внимание: Сезонный набо параметров P/D/Q не равен нулю, но m=0. Установка m=1.")
             m_val = 1

        seasonal_order = (P_val, D_val, Q_val, m_val)
        return {
            'order': order,
            'seasonal_order': seasonal_order
        }


class HybridConfigDialog(QDialog):
    """Диалоговое окно для настройки параметров гибридной модели SARIMA-LSTM."""
    def __init__(self, current_m=1, parent=None): 
        super().__init__(parent)
        self.setWindowTitle("Настройка SARIMA-LSTM") 
        layout = QFormLayout(self)

        
        sarima_label = QLabel("Параметры SARIMA:")
        sarima_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addRow(sarima_label)

        p_label = QLabel("Несезонный порядок (p, d, q):") 
        p_label.setToolTip("Авторегрессионный (p), Интегрированный (d), Скользящего среднего (q)")
        layout.addRow(p_label)
        order_layout = QHBoxLayout()
        self.p_spin = QSpinBox()
        self.p_spin.setRange(0, 10); self.p_spin.setValue(1)
        self.d_spin = QSpinBox()
        self.d_spin.setRange(0, 5); self.d_spin.setValue(1)
        self.q_spin = QSpinBox()
        self.q_spin.setRange(0, 10); self.q_spin.setValue(1)
        order_layout.addWidget(QLabel(" p:"))
        order_layout.addWidget(self.p_spin)
        order_layout.addWidget(QLabel(" d:"))
        order_layout.addWidget(self.d_spin)
        order_layout.addWidget(QLabel(" q:"))
        order_layout.addWidget(self.q_spin)
        layout.addRow(order_layout)

        P_label = QLabel("Сезонный порядок (P, D, Q, m):") 
        P_label.setToolTip("Сезонные: Авторегрессионный (P), Интегрированный (D), Скользящего среднего (Q), Период (m)")
        layout.addRow(P_label)
        seasonal_layout = QHBoxLayout()
        self.P_spin = QSpinBox()
        self.P_spin.setRange(0, 10); self.P_spin.setValue(1)
        self.D_spin = QSpinBox()
        self.D_spin.setRange(0, 5); self.D_spin.setValue(1)
        self.Q_spin = QSpinBox()
        self.Q_spin.setRange(0, 10); self.Q_spin.setValue(1)
        self.m_spin = QSpinBox()
        self.m_spin.setRange(0, 366); self.m_spin.setValue(max(0, current_m)) 
        seasonal_layout.addWidget(QLabel(" P:"))
        seasonal_layout.addWidget(self.P_spin)
        seasonal_layout.addWidget(QLabel(" D:"))
        seasonal_layout.addWidget(self.D_spin)
        seasonal_layout.addWidget(QLabel(" Q:"))
        seasonal_layout.addWidget(self.Q_spin)
        seasonal_layout.addWidget(QLabel(" m:"))
        seasonal_layout.addWidget(self.m_spin)
        layout.addRow(seasonal_layout)

        
        lstm_label = QLabel("Параметры LSTM (для остатков):") 
        lstm_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
        layout.addRow(lstm_label)

        self.sequence_len_spin = QSpinBox()
        self.sequence_len_spin.setRange(1, 100)
        self.sequence_len_spin.setValue(5)
        self.sequence_len_spin.setToolTip("Сколько предыдущих остатков использовать для предсказания следующего") 
        layout.addRow("Длина посл-ти остатков:", self.sequence_len_spin) 

        self.lstm_units_spin = QSpinBox()
        self.lstm_units_spin.setRange(10, 200)
        self.lstm_units_spin.setSingleStep(10)
        self.lstm_units_spin.setValue(50)
        self.lstm_units_spin.setToolTip("Количество нейронов в LSTM слое для остатков") 
        layout.addRow("Количество LSTM юнитов:", self.lstm_units_spin) 

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setToolTip("Количество эпох обучения LSTM на остатках") 
        layout.addRow("Количество эпох:", self.epochs_spin) 

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(32)
        self.batch_size_spin.setToolTip("Размер батча для обучения LSTM на остатках") 
        layout.addRow("Размер батча:", self.batch_size_spin) 

        
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addRow(self.buttonBox)

    def get_params(self):
        """Возвращает объединенные параметры в виде словаря."""
        
        sarima_order = (self.p_spin.value(), self.d_spin.value(), self.q_spin.value())
        m_val = self.m_spin.value()
        P_val = self.P_spin.value()
        D_val = self.D_spin.value()
        Q_val = self.Q_spin.value()
        if P_val == 0 and D_val == 0 and Q_val == 0:
            m_val = 0
        elif m_val == 0:
             print("Warning (Hybrid): Seasonal order P/D/Q is non-zero but m=0. Setting m=1 for seasonal order.")
             m_val = 1
        sarima_seasonal_order = (P_val, D_val, Q_val, m_val)

        
        lstm_params = {
            'sequence_len': self.sequence_len_spin.value(),
            'lstm_units': self.lstm_units_spin.value(),
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value()
        }

        return {
            'sarima_order': sarima_order,
            'sarima_seasonal_order': sarima_seasonal_order,
            'lstm_params': lstm_params 
        }



class MainWindow(QWidget):
    """Главное окно приложения для анализа и прогнозирования временных рядов."""
    def __init__(self):
        
        super().__init__()
        self.setWindowTitle("Автоматизация временных рядов") 
        self.resize(1000, 700)

        
        self.raw_df = None # Исходный загруженный фрейм данных
        self.current_df = None # текуие данные  
        self.train_df = None # Разделение обучающих данных
        self.test_df = None # Разделение тестовых данных
        self.date_col = None # Название столбца даты 
        self.value_col = None # Название столбца значений для прогнозирования
        self.frequency = None # Строка частот
        self.Seasonality = 52 # Период сезонности
        self.collapsed_element_height = 40  # Желаемая высота элементов в свернутом состоянии (например, 35px)
        self.collapsed_icon_size = QSize(24, 24)  # Желаемый размер иконок в свернутом состоянии (например, 22x22 px)
        self.original_icon_sizes = {}  # Словарь для хранения оригинальных размеров иконок кнопок

        # --- Main layout ---
        self.layout = QHBoxLayout(self) 

        # Sidebar
        self.sidebar_expanded = True
        self.sidebar_max_width = 350
        self.sidebar_min_width = 120
        self.sidebar = self.create_sidebar()
        self.layout.addWidget(self.sidebar)

        # Main area
        self.main_area = QVBoxLayout()
        self.dragdrop_widget = DragDropWidget(self.load_csv) 
        self.main_area.addWidget(self.dragdrop_widget)

        # Многоуровневый виджет для графиков и таблиц
        self.stack = QStackedWidget()
        self.stack.hide() # Изначально скрыт до тех пор, пока не будут загружены данные
        self.plot_canvas = QWebEngineView() # Для отображения Pлотных графиков
        self.stack.addWidget(self.plot_canvas) 
        self.table_widget = QTableWidget() # Таблица
        self.stack.addWidget(self.table_widget)  
        self.stack.setCurrentIndex(0) 
        self.table_widget.setSortingEnabled(True) 
        self.main_area.addWidget(self.stack)

        
        self.container = QWidget()
        self.container.setLayout(self.main_area)
        self.layout.addWidget(self.container)

        # Загрузка стилей из JSON
        try:
            style_sheet = load_styles_from_json("styles.json")
            self.setStyleSheet(style_sheet) 
        except FileNotFoundError:
            print("styles.json not found, using default styles.")
            

        self._disable_controls()


    def create_sidebar(self):
        
        
        frame = QFrame()
        frame.setMaximumWidth(self.sidebar_max_width)
        frame.setObjectName("SidebarFrame") #

        sidebar_content_layout = QVBoxLayout()
        sidebar_content_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Выровнение виджетов по верхнему краю, используя перечисление AlignmentFlag

        # --- 1. Main Menu ---
        main_menu = QMenu(self) 
        load_action = main_menu.addAction("Загрузить CSV") 
        clear_action = main_menu.addAction("Очистить все")
        toggle_view_action = main_menu.addAction("Переключить график/таблицу") 

        load_action.triggered.connect(self.load_csv_from_dialog)
        clear_action.triggered.connect(self.clear_all)
        toggle_view_action.triggered.connect(self.toggle_view)

        self.main_menu_btn = QPushButton("Файл и Вид") 
        self.main_menu_btn.setMenu(main_menu)
        qIcon = QIcon("icons/icons8_content.svg")
        self.main_menu_btn.setIcon(qIcon)
        self.original_icon_sizes[self.main_menu_btn] = self.main_menu_btn.iconSize() 
        self.main_menu_btn.setLayoutDirection(Qt.LeftToRight)

        self.main_menu_btn.setStyleSheet("""
                                QPushButton::menu-indicator {
                                image: url(icons/icons8_plus_math.svg);
                                subcontrol-origin: padding;
                                subcontrol-position: bottom right;
                                width: 18px;
                                height: 18px;}""")
        sidebar_content_layout.addWidget(self.main_menu_btn) 

        # --- 2. Data Settings ---
        settings_label = QLabel("Настройки данных:") 
        settings_label.setStyleSheet("font-weight: bold; margin-top: 10px;") # 
        sidebar_content_layout.addWidget(settings_label)

        self.date_col_label = QLabel("Столбец даты/времени:") 
        self.date_col_combo = QComboBox()
        self.date_col_combo.setStyleSheet("""
                                            QComboBox::drop-down {
                                                subcontrol-origin: padding;
                                                subcontrol-position: bottom right;
                                                width none;
                                                border: none;
                                            }

                                            QComboBox::down-arrow {
                                                image: url(icons/icons8_plus_math.svg);
                                                width: 16px;
                                                height: 16px;
                                            }""")
        self.date_col_combo.setToolTip("Выберите столбец с датами")
        sidebar_content_layout.addWidget(self.date_col_label)
        sidebar_content_layout.addWidget(self.date_col_combo) 

        self.value_col_label = QLabel("Столбец значения:")
        self.value_col_combo = QComboBox()
        self.value_col_combo.setStyleSheet("""
                                            QComboBox::drop-down {
                                                subcontrol-origin: padding;
                                                subcontrol-position: bottom right;
                                                width none;
                                                border: none;
                                            }

                                            QComboBox::down-arrow {
                                                image: url(icons/icons8_plus_math.svg);
                                                width: 16px;
                                                height: 16px;
                                            }""")

        self.value_col_combo.setToolTip("Выберите столбец с прогнозируемыми значениями") 
        
        sidebar_content_layout.addWidget(self.value_col_label)
        sidebar_content_layout.addWidget(self.value_col_combo) 

        self.freq_label = QLabel("Частота данных:") 
        self.freq_combo = QComboBox()
        self.freq_combo.setStyleSheet("""
                                    QComboBox::drop-down {
                                        subcontrol-origin: padding;
                                        subcontrol-position: bottom right;
                                        width none;
                                        border: none;
                                    }

                                    QComboBox::down-arrow {
                                        image: url(icons/icons8_plus_math.svg);
                                        width: 16px;
                                        height: 16px;
                                    }""")
       
        self.freq_combo.addItems(['D', 'B', 'W', 'M', 'Q', 'Y']) # Day, Business Day, Week, Month, Quarter, Year
        self.freq_combo.setToolTip("Выберите частоту временного ряда (D - день, M - месяц и т.д.)") 
        sidebar_content_layout.addWidget(self.freq_label)
        sidebar_content_layout.addWidget(self.freq_combo)

        self.apply_settings_btn = QPushButton("Применить настройки данных")
        
        qIcon_setting = QIcon("icons/icons8_settings.svg")
        self.apply_settings_btn.setIcon(qIcon_setting)
        self.original_icon_sizes[self.apply_settings_btn] = self.apply_settings_btn.iconSize() 
        self.apply_settings_btn.setLayoutDirection(Qt.LeftToRight)
        self.apply_settings_btn.clicked.connect(self.apply_data_settings)
        self.apply_settings_btn.setToolTip("Применить выбранные столбцы и частоту")
        sidebar_content_layout.addWidget(self.apply_settings_btn)

        # --- 3. Data Splitting ---
        split_label = QLabel("Разделение на выборки:") 
        split_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        sidebar_content_layout.addWidget(split_label)

        self.test_size_label = QLabel("Размер тестовой выборки (%):") 
        self.test_size_spinbox = QSpinBox()
        self.test_size_spinbox.setRange(5, 50) 
        self.test_size_spinbox.setValue(20) 
        self.test_size_spinbox.setToolTip("Укажите процент данных для тестовой выборки")
        sidebar_content_layout.addWidget(self.test_size_label)
        sidebar_content_layout.addWidget(self.test_size_spinbox) 

        self.split_data_btn = QPushButton("Разделить данные") 
        
        qIcon_split = QIcon("icons/updated_icons8_separate_document_1.svg")
        self.split_data_btn.setIcon(qIcon_split)
        self.original_icon_sizes[self.split_data_btn] = self.apply_settings_btn.iconSize()
        self.split_data_btn.setLayoutDirection(Qt.LeftToRight)
        self.split_data_btn.clicked.connect(self.split_data)
        self.split_data_btn.setToolTip("Разделить данные на обучающую и тестовую выборки") 
        sidebar_content_layout.addWidget(self.split_data_btn)

        # --- 4. Models Menu ---
        models_label = QLabel("Модели прогнозирования:") 
        models_label.setStyleSheet("font-weight: bold; margin-top: 10px;") 
        sidebar_content_layout.addWidget(models_label)

        # 4.1 Quantitative Methods
        quant_menu = QMenu(self)
        holt_winters_action = quant_menu.addAction("Экспоненциальное сглаживание (Holt-Winters)")
        sarima_action = quant_menu.addAction("SARIMA") 
      

        holt_winters_action.triggered.connect(self.holt_winters_prediction)
        
        sarima_action.triggered.connect(self.show_sarima_config_dialog)
        

        self.quant_menu_btn = QPushButton("Количественные методы")
        qIcon_kol_med = QIcon("icons/icons8_quantity.svg")
        self.quant_menu_btn.setIcon(qIcon_kol_med)
        self.original_icon_sizes[self.quant_menu_btn] = self.apply_settings_btn.iconSize() 

        self.quant_menu_btn.setLayoutDirection(Qt.LeftToRight)
        self.quant_menu_btn.setStyleSheet("""
                                QPushButton::menu-indicator {
                                image: url(icons/icons8_plus_math.svg);
                                subcontrol-origin: padding;
                                subcontrol-position: bottom right;
                                width: 18px;
                                height: 18px;}""")
        self.quant_menu_btn.setMenu(quant_menu)
        sidebar_content_layout.addWidget(self.quant_menu_btn)

        # 4.2 Neural Models
        nn_menu = QMenu(self)
        lstm_action = nn_menu.addAction("LSTM") 
        xgboost_action = nn_menu.addAction("XGBoost") 
        prophet_action = nn_menu.addAction("Prophet") 
        
        lstm_action.triggered.connect(self.show_lstm_config_dialog)
        xgboost_action.triggered.connect(self.show_xgboost_config_dialog) 
        prophet_action.triggered.connect(self.prophet_prediction)
        
        self.nn_menu_btn = QPushButton("Модели машиннго обучения") 
        qIcon_mash_med = QIcon("icons/icons8ML.svg")
        self.nn_menu_btn.setIcon(qIcon_mash_med)
        self.original_icon_sizes[self.nn_menu_btn] = self.apply_settings_btn.iconSize() 

        self.nn_menu_btn.setLayoutDirection(Qt.LeftToRight)
        self.nn_menu_btn.setStyleSheet("""
                                QPushButton::menu-indicator {
                                image: url(icons/icons8_plus_math.svg);
                                subcontrol-origin: padding;
                                subcontrol-position: bottom right;
                                width: 18px;
                                height: 18px;}""")
        self.nn_menu_btn.setMenu(nn_menu)
        sidebar_content_layout.addWidget(self.nn_menu_btn) 

        # 4.3 Hybrid Models
        hybrid_menu = QMenu(self) 
        sarima_lstm_action = hybrid_menu.addAction("SARIMA-LSTM")
        sarima_lstm_action.triggered.connect(self.show_hybrid_config_dialog) 
       


        self.hybrid_menu_btn = QPushButton("Гибридные модели")
        qIcon_gib_med = QIcon("icons/icons8_gibrid.svg")
        self.hybrid_menu_btn.setIcon(qIcon_gib_med)
        self.original_icon_sizes[self.hybrid_menu_btn] = self.apply_settings_btn.iconSize() 

        self.hybrid_menu_btn.setLayoutDirection(Qt.LeftToRight)
        self.hybrid_menu_btn.setStyleSheet("""
                                QPushButton::menu-indicator {
                                image: url(icons/icons8_plus_math.svg);
                                subcontrol-origin: padding;
                                subcontrol-position: bottom right;
                                width: 18px;
                                height: 18px;}""")
        self.hybrid_menu_btn.setMenu(hybrid_menu)
        sidebar_content_layout.addWidget(self.hybrid_menu_btn)



        # --- Spacer and Toggle Button ---
        sidebar_content_layout.addStretch(1) # Опускает кнопку переключения в нижнюю часть

        sidebar_content_widget = QWidget()
        sidebar_content_widget.setLayout(sidebar_content_layout)
        sidebar_content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred) 
        sidebar_content_widget.setObjectName("SidebarContent") # For styling

        horizontal_layout = QHBoxLayout() # Макет для размещения содержимого и кнопки переключения 
        horizontal_layout.setContentsMargins(0, 0, 0, 0)
        horizontal_layout.setSpacing(0)
        horizontal_layout.addWidget(sidebar_content_widget)

        self.toggle_button = QPushButton("<") # Кнопка сворачивания/разворачивания боковой панели
        self.toggle_button.setFixedWidth(15)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding) 
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #D9E6DA; /* Светло-серый полупрозрачный */
                color: #333333; /* Темный текст */
                border: none;
                font-size: 16px; /* Оставляем без изменений */
                border-left: 1px solid #cccccc; /* Светло-серая граница слева */
            }
            QPushButton:hover { background-color: #E8F5E9; } /* Светло-голубой при наведении */
        """)
        self.toggle_button.clicked.connect(self.toggle_sidebar)
        horizontal_layout.addWidget(self.toggle_button)

        frame.setLayout(horizontal_layout)
        return frame

    def toggle_sidebar(self):
        """Анимирует расширение или сворачивание боковой панели."""
        animation = QPropertyAnimation(self.sidebar, b"maximumWidth")
        animation.setDuration(300) # Продолжительность анимации в миллисекундах
        animation.setEasingCurve(QEasingCurve.Type.InOutQuart) # Анимационная кривая, используйте тип enum 

        if self.sidebar_expanded: 
            # Свернуто

            start_value = self.sidebar.width()
            end_value = self.sidebar_min_width
            self.toggle_button.setText(">")
            self._toggle_sidebar_elements_visibility(False) 
            # При необходимости обновите положение текста кнопки после анимации 
            animation.finished.connect(lambda: self._update_button_text_positions(False))
        else:
            # Развернуто
            start_value = self.sidebar.width()
            end_value = self.sidebar_max_width
            self.toggle_button.setText("<")
            self._toggle_sidebar_elements_visibility(True) 
            # При необходимости обновите положение текста кнопки после анимации  
            animation.finished.connect(lambda: self._update_button_text_positions(True))

        animation.setStartValue(start_value)
        animation.setEndValue(end_value) 

        self.sidebar_expanded = not self.sidebar_expanded
        animation.start()
        self.sidebar_animation = animation # Сохраняет ссылку, чтобы предотвратить сборку мусора

    def _toggle_sidebar_elements_visibility(self, visible):
        """Отображает или скрывает элементы управления на боковой панели, за исключением кнопки переключения.""" # 
        # Поиск основного вертикального расположения содержимого боковой панели
        content_widget = self.sidebar.findChild(QWidget, "SidebarContent") #  по названию объекта
        if content_widget:
             sidebar_layout = content_widget.layout() #Получить макет виджета содержимого
             if sidebar_layout:
                for i in range(sidebar_layout.count()):
                                item = sidebar_layout.itemAt(i)
                                widget = item.widget() if item else None
                                if widget and widget != self.toggle_button:
                                    if isinstance(widget, (QPushButton, QComboBox)):
                                        widget.setVisible(True)  # Кнопки и комбобоксы всегда видимы
                                    elif widget == self.test_size_spinbox: # Специально обрабатываем self.test_size_spinbox
                                        widget.setVisible(True)  # Всегда видимый
                                    elif isinstance(widget, QLabel) and widget == self.test_size_label: # Если хотите, чтобы и метка была видна
                                         widget.setVisible(False) # Оставить видимой или widget.setVisible(visible) для скрытия текста метки
                                    elif isinstance(widget, (QLabel, QSpinBox, QDoubleSpinBox)):
                                        # Остальные метки и спинбоксы (кроме test_size_spinbox)
                                        widget.setVisible(visible)

                        # Другие типы виджетов (если есть) будут вести себя по умолчанию)

    # или 
    def _update_button_text_positions(self, expanded):
        """Обновляет текст, высоту и размеры значков элементов боковой панели в зависимости от состояния."""
        


        # Исходная логика настройки текста для других элементов
        self.main_menu_btn.setText("Файл и Вид" if expanded else "") 
        self.quant_menu_btn.setText("Количественные методы" if expanded else "") 
        self.nn_menu_btn.setText("Модели машиннго обучения" if expanded else "") 
        self.hybrid_menu_btn.setText("Гибридные модели" if expanded else "") 
        self.apply_settings_btn.setText("Применить настройки данных" if expanded else "") 
        self.split_data_btn.setText("Разделить данные" if expanded else "") 
        
        widgets_to_adjust = [
            self.main_menu_btn, self.quant_menu_btn, self.nn_menu_btn, self.hybrid_menu_btn,
            self.apply_settings_btn, self.split_data_btn,
            self.date_col_combo, self.value_col_combo, self.freq_combo, 
            self.test_size_spinbox 
        ]

        if expanded:
            for widget in widgets_to_adjust:
                # Сброс высоты для динамической калибровки
                widget.setMinimumHeight(0) 
                widget.setMaximumHeight(16777215) # Максимальный размер
                
                if isinstance(widget, QPushButton) and widget in self.original_icon_sizes:
                    widget.setIconSize(self.original_icon_sizes[widget]) 
                
                widget.updateGeometry() # Запрос обновление геометрии 
                # widget.adjustSize() # Позволяет виджету соответствовать размеру, если это необходимо
        else: # Свернутое состояние для столбцов
            #self.date_col_combo.setCurrentIndex(-1)
            #self.value_col_combo.setCurrentIndex(-1)
            #self.date_col_combo.setStyleSheet("""QComboBox {color: #ff0000} """)

            for widget in widgets_to_adjust:
                widget.setFixedHeight(self.collapsed_element_height) 
                if isinstance(widget, QPushButton):
                    widget.setIconSize(self.collapsed_icon_size) 

                # Для QComboBox и QSpinBox, их внутренние иконки (стрелки)
                # обычно управляются стилями и должны адаптироваться или оставаться
                # фиксированными согласно стилю. setFixedHeight должно быть достаточно.
                # Ширина будет управляться общей шириной свернутой панели (self.sidebar_min_width)
                # и размеров виджетов.

    # ... (Продолжаем вырубать неугодные поля _disable_controls, _enable_..., clear_all, load_csv, load_csv_from_dialog, _populate..., apply_data_settings, split_data)
    def _disable_controls(self, disable_all=True):
        """Отключает элементы управления анализом. Можно отключить все элементы управления или только элементы управления после настройки данных.""" 
        if disable_all:
             # Отключить элементы управления выбором данных
             self.date_col_combo.setEnabled(False) 
             self.value_col_combo.setEnabled(False)
             self.freq_combo.setEnabled(False)
             self.apply_settings_btn.setEnabled(False)

        # Всегда отключаем элементы управления разделением и моделированием при вызове
        self.test_size_spinbox.setEnabled(False)
        self.split_data_btn.setEnabled(False)
        self.quant_menu_btn.setEnabled(False)
        self.nn_menu_btn.setEnabled(False) 
        self.hybrid_menu_btn.setEnabled(False)

    def _enable_data_settings_controls(self):
        """Включает элементы управления для выбора столбцов данных и частоты их использования."""
        self.date_col_combo.setEnabled(True) 
        self.value_col_combo.setEnabled(True)
        self.freq_combo.setEnabled(True)
        self.apply_settings_btn.setEnabled(True) 

    def _enable_analysis_controls(self):
        """Позволяет управлять разделением данных и выбором модели."""
        self.test_size_spinbox.setEnabled(True)
        self.split_data_btn.setEnabled(True)
        self.quant_menu_btn.setEnabled(True)
        self.nn_menu_btn.setEnabled(True) 
        self.hybrid_menu_btn.setEnabled(True)

    def clear_all(self):
        """Возвращает все данные, элементы управления и представление в исходное состояние."""
        # Сброс атрибутов данных 
        self.raw_df = None
        self.current_df = None
        self.train_df = None
        self.test_df = None
        self.date_col = None
        self.value_col = None
        self.frequency = None
        self.Seasonality = 52 

        self.plot_canvas.setHtml("") # Очистите вид графика 
        self.table_widget.clear() # Очистите содержимое таблицы
        self.table_widget.setRowCount(0) 
        self.table_widget.setColumnCount(0)
        self.stack.hide() # Спраятать стак (plot/table area)
        self.dragdrop_widget.show() # Снова отобразите область перетаскивания

        # Очистить и отключить элементы управления
        self.date_col_combo.clear() 
        self.value_col_combo.clear()
        self._disable_controls() 
        self.setWindowTitle("Автоматизация временных рядов")  #Сбросить заголовок окна 

    def load_csv(self, file_path):
        """Загружает данные из CSV-файла, заполняет поля выбора столбцов и отображает необработанные данные."""
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                 QMessageBox.warning(self, "Ошибка загрузки", "Выбранный CSV файл пуст.") 
                 return 

            self.clear_all() # Очистить любое предыдущее состояние
            self.raw_df = df.copy() # Сохранить исходный фрейм данных

            # Заполнить выпадающие списки выбора столбцов
            self._populate_column_selectors(list(df.columns)) # 

            # Отобразите необработанные данные в таблице
            self.display_table(self.raw_df) # Показать все столбцы из исходного файла
            self.stack.setCurrentWidget(self.table_widget) # Сначала показать таблицу
            self.stack.show() # отобразить стак
            self.dragdrop_widget.hide() # Скрыть виджет перетаскивания

            # Включить элементы управления для настроек данных
            self._enable_data_settings_controls() 

            QMessageBox.information(self, "Данные загружены",
                                      "Данные успешно загружены.\n"
                                      "Пожалуйста, выберите столбец даты, столбец значения и частоту данных, " 
                                     "затем нажмите 'Применить настройки данных'.")

        except Exception as e:
            self.clear_all() 
            QMessageBox.critical(self, "Ошибка загрузки CSV", f"Не удалось загрузить файл: {file_path}\nОшибка: {str(e)}") 

    def load_csv_from_dialog(self):
        """Открывает диалоговое окно с файлом, чтобы выбрать CSV-файл и загрузить его."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV", "", "CSV Files (*.csv)") 
        if file_path:
            self.load_csv(file_path) #

    def _populate_column_selectors(self, columns):
        """Заполняет поля со списком для выбора столбцов даты и значения.""" 
        self.date_col_combo.clear()
        self.value_col_combo.clear()
        self.date_col_combo.addItems(columns) 
        self.value_col_combo.addItems(columns)
        # Попытка угадать столбцы по умолчанию (optional)
        if 'date' in columns: 
             self.date_col_combo.setCurrentText('date')
        if r'Визиты' in columns: 
             self.value_col_combo.setCurrentText(r'Визиты')

    def apply_data_settings(self):
        """Применяет выбранные столбцы и частоту, обрабатывает данные и обновляет представление."""
        if self.raw_df is None:
             QMessageBox.warning(self, "Нет данных", "Сначала загрузите CSV файл.") 
             return 

        # Получить выбранные имена столбцов и частоту их использования
        self.date_col = self.date_col_combo.currentText()
        self.value_col = self.value_col_combo.currentText()
        self.frequency = self.freq_combo.currentText()

       
        if not self.date_col or not self.value_col:
            QMessageBox.warning(self, "Не выбраны столбцы", "Пожалуйста, выберите столбец даты и столбец значения.") 
            return 
        if self.date_col == self.value_col:
            QMessageBox.warning(self, "Ошибка выбора", "Столбец даты и столбец значения должны быть разными.")
            return

        try:
            # Создание рабочей копии, содержащую только выбранные столбцы 
            df = self.raw_df[[self.date_col, self.value_col]].copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col]) 
            df = df.set_index(self.date_col)

            df = df.asfreq(self.frequency)


            if df[self.value_col].isnull().any():
                 # Спросите пользователя, хочет ли он интерполировать пропущенные значения
                 reply = QMessageBox.question(self, "Обнаружены пропуски", 
                                           f"После установки частоты '{self.frequency}' появились пропуски в данных " 
                                           f"(возможно, из-за нерегулярных дат в исходном файле).\n" 
                                           f"Попытаться заполнить их линейной интерполяцией?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes) 
                 if reply == QMessageBox.StandardButton.Yes: 
                     df[self.value_col] = df[self.value_col].interpolate(method='linear')
                     
                     if df[self.value_col].isnull().any():
                         # Заполнение всех оставшиеся NAN, используя прямую и обратную заливку
                         df = df.fillna(method='bfill').fillna(method='ffill')
                 else:
                     QMessageBox.information(self, "Отмена", "Настройки не применены из-за пропусков.")
                     return 

            
            if df[self.value_col].isnull().any():
                 QMessageBox.warning(self, "Ошибка", "Не удалось заполнить все пропуски. Настройки не применены.")
                 return 

            # Сохранение обработаного фрейма данных
            self.current_df = df.copy()
            # Сброс разделений между тренировками и тестами при изменении настроек данных
            self.train_df = None
            self.test_df = None
             # Обновлять сезонность в зависимости от частоты
            self._update_default_seasonality()


            # Обновите таблицу и постройте график с обработанными данными
            self.display_table(self.current_df.reset_index()) # Показать обработанные данные
            self.plot_data(self.current_df) 

            # Включить элементы управления анализом 
            self._enable_analysis_controls() 
            self.stack.show()
            self.dragdrop_widget.hide()

            QMessageBox.information(self, "Настройки применены", 
                                    f"Данные обработаны:\n"
                                    f"- Столбец даты: {self.date_col}\n" 
                                    f"- Столбец значения: {self.value_col}\n" 
                                    f"- Частота: {self.frequency}\n"
                                    f"Теперь можно разделить данные или применить модель.") 

        except Exception as e:
            QMessageBox.critical(self, "Ошибка обработки данных", f"Не удалось применить настройки.\nОшибка: {str(e)}\n{traceback.format_exc()}")
            
            self._disable_controls(disable_all=False)

    def _update_default_seasonality(self):
        """Определяеv период сезонности по умолчанию на основе частоты передачи данных."""
        if self.frequency:
            freq_upper = self.frequency.upper()
            if 'D' in freq_upper: # Daily
                self.Seasonality = 7
            elif 'W' in freq_upper: # Weekly
                self.Seasonality = 52
            elif 'M' in freq_upper: # Monthly
                self.Seasonality = 12
            elif 'Q' in freq_upper: # Quarterly
                self.Seasonality = 4
            elif 'Y' in freq_upper or 'A' in freq_upper: # Yearly
                self.Seasonality = 1 #без сезона
            else: 
                self.Seasonality = 5 
            print(f"Default seasonality 'm' updated to {self.Seasonality} based on frequency '{self.frequency}'.")
        else:
             self.Seasonality = 1 

    def split_data(self):
        """Разбиваем current_df на train_df и test_df в зависимости от выбранного процента."""
         
        if self.current_df is None or self.value_col is None: 
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите и обработайте данные.") 
            return 

        test_percentage = self.test_size_spinbox.value() / 100.0
        if not (0 < test_percentage < 1):
             QMessageBox.warning(self, "Неверный процент", "Размер тестовой выборки должен быть между 0% и 100%.")
             return 

       
        split_index = int(len(self.current_df) * (1 - test_percentage))

        if split_index < 1 or split_index >= len(self.current_df):
             QMessageBox.warning(self, "Ошибка разделения", "Недостаточно данных для указанного размера тестовой выборки.") 
             return

       
        self.train_df = self.current_df.iloc[:split_index] 
        self.test_df = self.current_df.iloc[split_index:]

        QMessageBox.information(self, "Данные разделены", 
                                 f"Данные разделены:\n" 
                                 f"- Обучающая выборка: {len(self.train_df)} точек\n" 
                                 f"- Тестовая выборка: {len(self.test_df)} точек") 

        
        self.plot_data(self.current_df, split_point=self.train_df.index[-1]) 

    def plot_data(self, df_to_plot, forecast_df=None, title_suffix="", split_point=None):
         """Отображает данные и опционально прогноз и точку разделения."""
         if df_to_plot is None or self.value_col is None:
             return # Нечего рисовать

         try:
             fig = go.Figure() # Используем go.Figure для большей гибкости 

             # --- 1. Обучающие данные ---
             plot_train_df = self.train_df if self.train_df is not None else df_to_plot
             last_train_point_series = None # Сохраним последнюю точку для соединения
             if plot_train_df is not None and not plot_train_df.empty:
                 fig.add_trace(go.Scatter(x=plot_train_df.index, y=plot_train_df[self.value_col],
                                        mode='lines', name='Исходные/Обучающие данные')) 
                 # Сохраняем последнюю точку обучающей выборки (как Series)
                 last_train_point_series = plot_train_df[[self.value_col]].iloc[[-1]]


             # --- 2. Тестовые данные (с соединением) ---
             if self.test_df is not None:
                 test_data_for_plot = self.test_df[[self.value_col]].copy()
                 plot_test_combined = test_data_for_plot # По умолчанию - только тестовые

                 if last_train_point_series is not None and not test_data_for_plot.empty:
                      # Объединяем последнюю точку train с началом test
                      plot_test_combined = pd.concat([last_train_point_series, test_data_for_plot])

                 if not plot_test_combined.empty:
                     fig.add_trace(go.Scatter(x=plot_test_combined.index, y=plot_test_combined[self.value_col],
                                           mode='lines', name='Тестовые данные',
                                           line=dict(color='gray'))) 


             # --- 3. Прогноз (с соединением fitted и forecast) ---
             last_fitted_point_series = None # Сохраним последнюю точку fitted
             if forecast_df is not None:

                 # 3.1 Прогноз на обучающей выборке (fitted values)
                 if 'fitted' in forecast_df.columns:
                      fitted_values_to_plot = forecast_df['fitted'].dropna()
                      if not fitted_values_to_plot.empty:
                           fig.add_trace(go.Scatter(x=fitted_values_to_plot.index, y=fitted_values_to_plot,
                                                   mode='lines', name='Прогноз на обуч.',
                                                   line=dict(color='orange', dash='dash'))) 
                           # Сохраняем последнюю точку fitted (как Series)
                           last_fitted_point_series = fitted_values_to_plot.iloc[[-1]].rename('forecast') # Переименуем для конкатенации

                 # 3.2 Прогноз на будущее/тестовый период
                 if 'forecast' in forecast_df.columns:
                      forecast_values_to_plot = forecast_df['forecast'].dropna()
                      plot_forecast_combined = forecast_values_to_plot # По умолчанию - только прогноз

                      if last_fitted_point_series is not None and not forecast_values_to_plot.empty:
                           # Объединяем последнюю точку fitted с началом forecast
                            # Убедимся, что last_fitted_point_series это Series с именем 'forecast'
                           last_fitted_point_df = pd.DataFrame(last_fitted_point_series)
                           forecast_values_df = pd.DataFrame(forecast_values_to_plot)
                           plot_forecast_combined = pd.concat([last_fitted_point_df, forecast_values_df])
                           # Уберем возможное дублирование индекса, если fitted и forecast пересекаются
                           plot_forecast_combined = plot_forecast_combined[~plot_forecast_combined.index.duplicated(keep='first')]


                      if not plot_forecast_combined.empty:
                           fig.add_trace(go.Scatter(x=plot_forecast_combined.index, y=plot_forecast_combined['forecast'],
                                                   mode='lines', name='Прогноз',
                                                   line=dict(color='red')))

                 # 3.3 Доверительные интервалы (без соединения)
                 if 'lower_ci' in forecast_df.columns and 'upper_ci' in forecast_df.columns:
                        # Берем все доступные CI (fitted + forecast), предполагая что они в одном DataFrame
                        ci_df = forecast_df[['lower_ci', 'upper_ci']].dropna()
                        if not ci_df.empty:
                            fig.add_trace(go.Scatter(x=ci_df.index, y=ci_df['upper_ci'],
                                                    mode='lines', line=dict(width=0),
                                                    showlegend=False)) # [ 75 ]
                            fig.add_trace(go.Scatter(x=ci_df.index, y=ci_df['lower_ci'],
                                                    mode='lines', line=dict(width=0),
                                                    fillcolor='rgba(255, 0, 0, 0.2)',
                                                    fill='tonexty', name='Дов. интервал')) 


             # --- 4. Вертикальная линия разделения ---
             if split_point is not None:
                 try:
                     # Определяем y_min и y_max для линии
                     all_data_for_y_range = [plot_train_df[self.value_col]]
                     if self.test_df is not None:
                         all_data_for_y_range.append(self.test_df[self.value_col])
                     if forecast_df is not None:
                          if 'fitted' in forecast_df: all_data_for_y_range.append(forecast_df['fitted'])
                          if 'forecast' in forecast_df: all_data_for_y_range.append(forecast_df['forecast'])
                          if 'lower_ci' in forecast_df: all_data_for_y_range.append(forecast_df['lower_ci'])
                          if 'upper_ci' in forecast_df: all_data_for_y_range.append(forecast_df['upper_ci'])

                     combined_y_values = pd.concat(all_data_for_y_range).dropna()
                     y_min = combined_y_values.min()
                     y_max = combined_y_values.max()
                     y_range = y_max - y_min
                     # Добавляем небольшой отступ сверху и снизу
                     y_plot_min = y_min - y_range * 0.05
                     y_plot_max = y_max + y_range * 0.05


                     # Добавляем линию [ 83 ]
                     fig.add_trace(go.Scatter(
                         x=[split_point, split_point],
                         y=[y_plot_min, y_plot_max], # Используем расчетный диапазон
                         mode='lines',
                         line=dict(color='green', width=2, dash='dash'), 
                         name='Разделение',
                         showlegend=True # Можно показать в легенде
                     ))
                     # Аннотацию можно убрать, если линия есть в легенде
                     

                 except Exception as line_err:
                      print(f"Предупреждение: Не удалось нарисовать линию разделения. Ошибка: {line_err}")


             # --- 5. Настройка макета и отображение ---
             title = f'Временной ряд: {self.value_col}{title_suffix}'
             fig.update_layout(
                 title=title,
                 template='plotly',
                 xaxis_title=self.date_col or "Дата",
                 yaxis_title=self.value_col or "Значение",
                 plot_bgcolor='#E8F5E9',
                 paper_bgcolor='#D9E6DA',
                 xaxis=dict(gridcolor='rgba(0,0,255,0.1)'),
                 yaxis=dict(gridcolor='rgba(0,0,255,0.1)')
             )
             
             html = fig.to_html(include_plotlyjs='cdn')
             background_color = "#FAFAFA"
             style_tag = f"<style>body {{ background-color: {background_color}; margin: 0; padding: 0; color: #333333; }}</style>"
             html = html.replace("</head>", f"{style_tag}</head>", 1)
             
             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
             with open(temp_file.name, 'w', encoding='utf-8') as f:
                 f.write(html)
             self.plot_canvas.load(QUrl.fromLocalFile(temp_file.name))
             self.stack.setCurrentWidget(self.plot_canvas)


         except Exception as e:
             QMessageBox.critical(self, "Ошибка построения графика", f"Не удалось построить график.\nОшибка: {str(e)}\n{traceback.format_exc()}")



    def display_table(self, df):
        """Отображает фрейм данных pandas в QTableWidget."""
        try: 
            self.table_widget.clear()
            if df is None:
                 self.table_widget.setRowCount(0)
                 self.table_widget.setColumnCount(0)
                 return

            df_display = df.copy()
            # Если индексом является DatetimeIndex, сбросить его, чтобы он стал столбцом для отображения
            if isinstance(df_display.index, pd.DatetimeIndex):
                 df_display.reset_index(inplace=True)
                 # Отформатируйте столбец даты для лучшей читаемости
                 date_col_name = df_display.columns[0] # Название прежнего индекса
                 try:  
                     df_display[date_col_name] = df_display[date_col_name].dt.strftime('%Y-%m-%d %H:%M:%S')#универсальный формат
                 except AttributeError:
                     pass 

            # Установим размеры и заголовки таблиц
            self.table_widget.setRowCount(df_display.shape[0])
            self.table_widget.setColumnCount(df_display.shape[1])
            # заголовки являются строками
            self.table_widget.setHorizontalHeaderLabels([str(col) for col in df_display.columns])

            # Заполнение ячеек таблицы
            for row in range(df_display.shape[0]):
                for col in range(df_display.shape[1]):
                    item_value = df_display.iloc[row, col]
                    # форматированиее чисел
                    if isinstance(item_value, (int, float, np.number)):
                         item_text = f"{item_value:.2f}" if isinstance(item_value, (float, np.floating)) else str(item_value)
                    else:
                         item_text = str(item_value)
                    item = QTableWidgetItem(item_text) 
                    self.table_widget.setItem(row, col, item)

            self.table_widget.resizeColumnsToContents() # Настройка ширины столбцов

        except Exception as e:
            QMessageBox.critical(self, "Ошибка отображения таблицы", f"Не удалось отобразить данные.\nОшибка: {str(e)}") 
            self.table_widget.clear()
            self.table_widget.setRowCount(0)
            self.table_widget.setColumnCount(0)

    def toggle_view(self):
        """Переключает представление между графиком и таблицей в QStackedWidget."""
        if not self.stack.isVisible():
             QMessageBox.information(self,"Информация","Нет данных для отображения.")
             return
        current_index = self.stack.currentIndex() 
        # Переключитесь на другой индекс
        new_index = 1 if current_index == 0 else 0
        self.stack.setCurrentIndex(new_index)


    # --- Функции прогнозирования ---


    def show_lstm_config_dialog(self):
        """Показывает диалоговое окно настройки LSTM и запускает прогнозирование, если оно принято."""
        # Проверка необходимых условий
        if self.current_df is None or self.value_col is None: 
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите и обработайте данные.") 
            return 
        if self.train_df is None or self.test_df is None:
             QMessageBox.warning(self, "Разделение необходимо", "Для обучения LSTM необходимо разделить данные на обучающую и тестовую выборки.")
             return 

        dialog = LSTMConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted: #
            params = dialog.get_params()
            self.lstm_prediction(params) # Функция прогнозирования вызовов с параметрами


    def lstm_prediction(self, params):
        """Выполняет прогнозирование LSTM с использованием предоставленных параметров с отслеживанием прогресса."""
        if self.train_df is None or self.test_df is None or self.value_col is None:
            QMessageBox.warning(self, "Нет данных", "Нет данных для обучения/тестирования LSTM.")
            return

        msg_box = None
        progress_dialog = None
        try:
            # --- 1. Data Preparation ---
            value_col = self.value_col
            data_train = self.train_df[[value_col]]
            data_test = self.test_df[[value_col]]

            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train_data = scaler.fit_transform(data_train)
            scaled_full_data = scaler.transform(self.current_df[[value_col]])


            sequence_len = params['sequence_len']
            if sequence_len <= 0:
                raise ValueError("Длина последовательности (sequence_len) должна быть > 0.")


            X_train, y_train = [], []
            if len(scaled_train_data) <= sequence_len:
                raise ValueError(f"Недостаточно данных ({len(scaled_train_data)}) для создания LSTM последовательности длиной {sequence_len}.")

            for i in range(sequence_len, len(scaled_train_data)):
                X_train.append(scaled_train_data[i-sequence_len:i, 0])
                y_train.append(scaled_train_data[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)

            # Изменяет форму X_train для ввода в LSTM
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # --- 2. Создание моделей и обучение ---
            model = Sequential()
            model.add(LSTM(units=params['lstm_units'], return_sequences=False,
                           input_shape=(X_train.shape[1], 1)))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # --- Настройка диалогового окна выполнения ---
            progress_dialog = QProgressDialog("Обучение LSTM...", "Cancel", 0, params['epochs'], self)
            progress_dialog.setWindowTitle("Обучение LSTM")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setWindowIcon(QIcon())  # Убираем иконку
            progress_dialog.setWindowFlags(progress_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # Убираем подсказку '?'
            progress_dialog.resize(250, 150)  # Устанавливаем размер окна
            progress_dialog.show()
            QApplication.processEvents()

            # Настраиваемый обратный вызов Keras для обновления диалогового окна хода выполнения
            class ProgressCallback(Callback):
                def __init__(self, progress_dialog, total_epochs):
                    super().__init__()
                    self.progress_dialog = progress_dialog
                    self.total_epochs = total_epochs

                def on_epoch_end(self, epoch, logs=None):
                    self.progress_dialog.setValue(epoch + 1)
                    QApplication.processEvents()
                    if self.progress_dialog.wasCanceled():
                        self.model.stop_training = True

            # Обучите модель с помощью обратного вызова хода выполнения
            history = None
            try:
                history = model.fit(X_train, y_train,
                                   epochs=params['epochs'],
                                   batch_size=params['batch_size'],
                                   verbose=0,
                                   callbacks=[ProgressCallback(progress_dialog, params['epochs'])])
            finally:
                #  диалоговое окно выполнения закрыто
                if progress_dialog is not None:
                    progress_dialog.close()
                QApplication.processEvents()
                if history is None:
                    raise Exception("LSTM training was not completed or was cancelled.")

            # --- 3. Предсказание ---
            input_start_index = len(scaled_full_data) - len(data_test) - sequence_len
            if input_start_index < 0:
                raise ValueError(f"Cannot create test sequences. Not enough data before the test set start. Need {sequence_len}, have {len(scaled_full_data) - len(data_test)}.")

            inputs = scaled_full_data[input_start_index:]
            X_test = []
            for i in range(sequence_len, len(inputs)):
                X_test.append(inputs[i-sequence_len:i, 0])

            if not X_test:
                raise ValueError("Could not create any test sequences (X_test). Check sequence_len and test set size.")

            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # Масштабированные прогнозы
            predictions_scaled = model.predict(X_test)
            # Предсказания в обратном преобразовании
            predictions = scaler.inverse_transform(predictions_scaled)

            # --- 4. Форматирование и отображение результатов ---
            raw_predictions = predictions.flatten()

            # --- Установка значения ---
            train_predictions_scaled = model.predict(X_train)
            train_predictions = scaler.inverse_transform(train_predictions_scaled).flatten()

            if sequence_len >= len(self.train_df.index):
                raise ValueError("sequence_len is too large for the training data size.")
            try:
                original_fitted_index = self.train_df.index[sequence_len:].shift(-2)
            except Exception as e:
                print(f"Warning: Could not shift fitted_index. Using original slice. Error: {e}")
                original_fitted_index = self.train_df.index[sequence_len:]

            if len(train_predictions) != len(original_fitted_index):
                print(f"Warning: Mismatch between train predictions ({len(train_predictions)}) and fitted index ({len(original_fitted_index)}). Adjusting index.")
                original_fitted_index = original_fitted_index[:len(train_predictions)]

            fitted_values_df_orig = pd.DataFrame({'fitted': train_predictions}, index=original_fitted_index)

            # --- Прогнозируемые значения ---
            try:
                original_forecast_index = self.test_df.index.shift(-2)
            except Exception as e:
                print(f"Warning: Could not shift forecast_index. Using original test_df index. Error: {e}")
                original_forecast_index = self.test_df.index

            if len(raw_predictions) != len(original_forecast_index):
                print(f"Warning: Mismatch between raw prediction length ({len(raw_predictions)}) and shifted test index length ({len(original_forecast_index)}). Truncating to minimum length.")
                min_len = min(len(raw_predictions), len(original_forecast_index))
                raw_predictions = raw_predictions[:min_len]
                original_forecast_index = original_forecast_index[:min_len]

            forecast_df_orig = pd.DataFrame({'forecast': raw_predictions}, index=original_forecast_index)

            # --- Приготовления к составлению графика ---
            if len(forecast_df_orig) >= 2:
                first_two_forecast = forecast_df_orig.iloc[:2].rename(columns={'forecast': 'fitted'})
                plot_fitted_df = pd.concat([fitted_values_df_orig, first_two_forecast])
                plot_forecast_df_part = forecast_df_orig.iloc[2:]
            else:
                print("Warning: Less than 2 forecast points generated. Plotting as is.")
                plot_fitted_df = fitted_values_df_orig
                plot_forecast_df_part = forecast_df_orig

            plot_combined_df = pd.concat([plot_fitted_df[['fitted']], plot_forecast_df_part[['forecast']]], axis=1)

            # --- График ---
            self.plot_data(self.current_df,
                           forecast_df=plot_combined_df,
                           title_suffix=" (LSTM)",
                           split_point=self.train_df.index[-1])

            # --- Расчет метрик ---
            y_true = self.test_df[self.value_col]
            y_pred_values = raw_predictions

            if len(y_true) != len(y_pred_values):
                print(f"Aligning y_true length ({len(y_true)}) to y_pred length ({len(y_pred_values)}) for metrics.")
                y_true = y_true.iloc[:len(y_pred_values)]

            if len(y_true) == len(y_pred_values) and len(y_true) > 0:
                self.calculate_and_display_metrics(y_true, y_pred_values, "LSTM")
            else:
                print("Warning: Cannot calculate metrics due to length mismatch or zero length after alignment.")
                QMessageBox.warning(self, "Ошибка метрик", "Не удалось рассчитать метрики из-за несоответствия данных.")

        except Exception as e:

            if msg_box is not None and msg_box.isVisible():
                msg_box.close()
            if progress_dialog is not None:
                progress_dialog.close()
            QMessageBox.critical(self, "Ошибка LSTM", f"Ошибка при прогнозировании LSTM: {str(e)}\n{traceback.format_exc()}")
            print("---- LSTM Error Traceback ----")
            traceback.print_exc()
            print("-----------------------------")


    def holt_winters_prediction(self):
        """Выполняет прогнозирование экспоненциального сглаживания по методу Холта-Уинтерса."""
        if self.current_df is None or self.value_col is None: 
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите, настройте и, возможно, разделите данные.") 
            return

        try:
            # Использует обучающие данные, если таковые имеются, в противном случае использует весь набор данных целиком
            data_to_fit = self.train_df if self.train_df is not None else self.current_df
            if data_to_fit is None or data_to_fit.empty:
                 QMessageBox.warning(self,"Нет данных для обучения", "Обучающая выборка пуста.") 
                 return 


            # --- Использует self.Seasonality (обновляет из _update_default_seasonality) ---
            seasonal_periods = self.Seasonality 
            trend_component = 'add' # or 'mul', None
            seasonal_component = 'add' # or 'mul', None

            # Определени количества шагов прогноза
            if self.test_df is not None: 
                 forecast_steps = len(self.test_df) # Прогноз на период тестового набора
            else:
                 # Прогноз на 10% вперед, если тест не установлен (с минимальными/максимальными ограничениями)
                 forecast_steps = min(10, max(1, int(len(data_to_fit) * 0.1))) 

            # Проверка, достаточно ли данных для указанной сезонности
            if seasonal_periods is not None and seasonal_periods > 0 and len(data_to_fit) < 2 * seasonal_periods: 
                 reply = QMessageBox.warning(self,"Мало данных для сезонности", 
                                           f"Недостаточно данных ({len(data_to_fit)}) для сезонности {seasonal_periods}.\n" 
                                           f"Попробовать модель без сезонности?", 
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes) 
                 if reply == QMessageBox.StandardButton.Yes: 
                     seasonal_periods = None 
                     seasonal_component = None
                 else:
                      return 
            elif seasonal_periods is None or seasonal_periods <= 0:
                 seasonal_periods = None 
                 seasonal_component = None 


            # обучение модель экспоненциального сглаживания
            model = ExponentialSmoothing(
                data_to_fit[self.value_col],
                seasonal=seasonal_component, # Сезонная составляющая, определяемая пропуском
                seasonal_periods=seasonal_periods, 
                trend=trend_component, 
                initialization_method='estimated' # statsmodels оценивает начальное состояние
            )
            fitted_model = model.fit()

            # Создаем прогноз
            forecast_values = fitted_model.forecast(steps=forecast_steps)

            # Создаем прогнозируемый фрейм данных с соответствующим индексом 
            if self.test_df is not None:
                 forecast_index = self.test_df.index # тест на индекс 
            else:
                 # Генерировать будущие даты, если прогнозирование выходит за рамки имеющихся данных
                 try: 
                     last_date = data_to_fit.index[-1] 
                     #  диапазон дат, начинающийся после последней известной даты
                     forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1 if self.frequency == 'D' else 0), 
                                                    periods=forecast_steps, freq=self.frequency) 
                 except Exception as date_err:
                     QMessageBox.critical(self, "Ошибка генерации индекса", f"Не удалось создать индекс для прогноза: {date_err}") 
                     return 

            forecast_df = pd.DataFrame({'forecast': forecast_values}, index=forecast_index)
            # Получить подобранные значения из модели
            fitted_values_df = pd.DataFrame({'fitted': fitted_model.fittedvalues}, index=data_to_fit.index)
            # комбинируем данные и прогноз для построения графика
            plot_forecast_df = pd.concat([fitted_values_df, forecast_df]) 

            # График
            self.plot_data(self.current_df, forecast_df=plot_forecast_df,
                           title_suffix=" (Holt-Winters)",
                           split_point=self.train_df.index[-1] if self.train_df is not None else None) 

            # Вычисляем и отображаем показатели при наличии тестовых данных
            if self.test_df is not None and not self.test_df.empty: 
                y_true = self.test_df[self.value_col]
                y_pred = forecast_df['forecast'] 
                self.calculate_and_display_metrics(y_true, y_pred, "Holt-Winters") 

            #QMessageBox.information(self, "Holt-Winters Завершено", "Прогнозирование завершено.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка Holt-Winters", f"Ошибка при прогнозировании: {str(e)}\n{traceback.format_exc()}") 



    def show_sarima_config_dialog(self):
        """Показывает диалоговое окно настройки SARIMA и запускает прогнозирование."""

        if self.current_df is None or self.value_col is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите и обработайте данные.")
            return
        
        if self.train_df is None:
             QMessageBox.warning(self, "Нет данных для обучения", "Для обучения SARIMA необходима обучающая выборка (разделите данные).") 
             return
        if self.train_df.empty:
             QMessageBox.warning(self,"Нет данных для обучения", "Обучающая выборка пуста.")
             return

        # Передаем текущую сезонность по умолчанию "m" в диалоговое окно
        dialog = SARIMAConfigDialog(current_m=self.Seasonality, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted: 
            params = dialog.get_params()
             
            new_m = params['seasonal_order'][3]
            if new_m > 0:
                 self.Seasonality = new_m
            self.sarima_prediction(params) 



    def sarima_prediction(self, params): 
        """Прогнозирование SARIMA с использованием предоставленных параметров."""
        if self.current_df is None or self.value_col is None:

            QMessageBox.warning(self, "Нет данных", "Сначала загрузите, настройте и, возможно, разделите данные.")
            return 

        msg_box = None
        try:
            data_to_fit = self.train_df
            if data_to_fit is None or data_to_fit.empty: #
                  QMessageBox.warning(self,"Нет данных для обучения", "Обучающая выборка пуста.") 
                  return 

            # --- Использует параметры ARIMA из диалогового окна ---
            order = params['order']
            seasonal_order = params['seasonal_order']
            seasonal_periods = seasonal_order[3] # 'm'
            # ---------------------------------------------

            # количество шагов прогноза
            if self.test_df is not None: 
                  forecast_steps = len(self.test_df) # Прогноз на период тестового набора 
            else:
                 # если тест не установлен
                 forecast_steps = min(10, max(1, int(len(data_to_fit) * 0.1))) 


            if seasonal_periods is not None and seasonal_periods > 0 and len(data_to_fit) < 2 * seasonal_periods:

                 QMessageBox.warning(self,"Мало данных для сезонности SARIMA",
                                          f"Недостаточно данных ({len(data_to_fit)}) для сезонности m={seasonal_periods}. Модель может не обучиться.")
                 pass # Разрешить попытку, но предупреждать



            # Отобразить сообщение о том, что подгонка модели продолжается
            msg_box = QMessageBox(self) 
            msg_box.setIcon(QMessageBox.Icon.Information) 
            msg_box.setText("Идет обучение модели SARIMA, это может занять некоторое время...") 
            msg_box.setWindowTitle("Обучение SARIMA") 
            msg_box.setStandardButtons(QMessageBox.StandardButton.Cancel) 
            msg_box.show()
            QApplication.processEvents() 

            # Обучение модели
            model = SARIMAX(
                 data_to_fit[self.value_col],
                 order=order, 
                 seasonal_order=seasonal_order, 
                 enforce_stationarity=False, 
                 enforce_invertibility=False 
            )

            fitted_model = None
            try:
                 fitted_model = model.fit(disp=False) 
            finally:
                if msg_box is not None and msg_box.isVisible(): 
                     msg_box.close()
                     QApplication.processEvents() 

            if fitted_model is None: 
                 raise Exception("Обучение SARIMA не было завершено.")

            # Получть результаты прогноза
            forecast_result = fitted_model.get_forecast(steps=forecast_steps) 
            forecast_values = forecast_result.predicted_mean 



            if self.test_df is not None:
                  forecast_index = self.test_df.index 
            else:
                try:
                    last_date = data_to_fit.index[-1]
                    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1 if self.frequency == 'D' else 0), periods=forecast_steps, freq=self.frequency)
                except Exception as date_err: 
                     QMessageBox.critical(self, "Ошибка генерации индекса", f"Не удалось создать индекс для прогноза: {date_err}") #
                     return 

            # фрейм данных для прогнозируемых значений
            forecast_df = pd.DataFrame({'forecast': forecast_values}, index=forecast_index) 
            fitted_values_df = pd.DataFrame({'fitted': fitted_model.fittedvalues}, index=data_to_fit.index)

            # Комбинация данных и прогноза для построения графика
            plot_forecast_df = pd.concat([fitted_values_df[['fitted']], forecast_df[['forecast']]])

            self.plot_data(self.current_df, forecast_df=plot_forecast_df,
                           title_suffix=" (SARIMA)", 
                           split_point=self.train_df.index[-1] if self.train_df is not None else None) 

            # Вычислять и отображать метрик при наличии тестовых данных
            if self.test_df is not None and not self.test_df.empty:
                y_true = self.test_df[self.value_col]
                y_pred = forecast_df['forecast'] 
                self.calculate_and_display_metrics(y_true, y_pred, "SARIMA") 

            #QMessageBox.information(self, "SARIMA Завершено", "Прогнозирование завершено.") 

        except Exception as e: 
            QMessageBox.critical(self, "Ошибка SARIMA", f"Ошибка при прогнозировании: {str(e)}\n{traceback.format_exc()}") 
        finally:
             if msg_box is not None and msg_box.isVisible(): 
                  msg_box.close()
                  QApplication.processEvents() 


    def show_hybrid_config_dialog(self):
        """Отображает диалоговое окно гибридной конфигурации SARIMA-LSTM и запускает прогнозирование."""
        if self.current_df is None or self.value_col is None: 
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите и обработайте данные.") 
            return 
        if self.train_df is None or self.test_df is None:
            QMessageBox.warning(self, "Разделение необходимо", "Для гибридной модели SARIMA-LSTM необходимо разделить данные на обучающую и тестовую выборки.") #
            return 


        dialog = HybridConfigDialog(current_m=self.Seasonality, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted: 
            params = dialog.get_params()
            new_m = params['sarima_seasonal_order'][3]
            if new_m > 0:
                 self.Seasonality = new_m
            self.sarima_lstm_prediction(params)  

    def sarima_lstm_prediction(self, params):
        """Выполняет гибридное прогнозирование SARIMA-LSTM с использованием параметров из диалогового окна с отслеживанием хода выполнения."""
        if self.train_df is None or self.test_df is None or self.value_col is None:
            QMessageBox.warning(self, "Нет данных", "Нет данных для обучения/тестирования SARIMA-LSTM.")
            return

        sarima_msg_box = None
        progress_dialog = None
        forecast_progress_dialog = None

        try:
            value_col = self.value_col
            train_data = self.train_df[[value_col]]
            test_data = self.test_df[[value_col]]
            forecast_steps = len(test_data)

            sarima_order = params['sarima_order']
            sarima_seasonal_order = params['sarima_seasonal_order']
            lstm_params = params['lstm_params']

            # --- Шаг 1: Обучение SARIMA и получить остатки ---
            sarima_msg_box = QMessageBox(self)
            sarima_msg_box.setIcon(QMessageBox.Icon.Information)
            sarima_msg_box.setText("Шаг 1/3: Обучение SARIMA...")
            sarima_msg_box.setWindowTitle("SARIMA-LSTM")
            sarima_msg_box.setStandardButtons(QMessageBox.StandardButton.Cancel)
            sarima_msg_box.show()
            QApplication.processEvents()

            seasonal_periods = sarima_seasonal_order[3]
            if seasonal_periods > 0 and len(train_data) < 2 * seasonal_periods:
                raise ValueError(f"Недостаточно данных ({len(train_data)}) для сезонности m={seasonal_periods} в SARIMA.")

            sarima_model = SARIMAX(
                train_data[value_col],
                order=sarima_order,
                seasonal_order=sarima_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarima_fitted = sarima_model.fit(disp=False)

            sarima_forecast = sarima_fitted.get_forecast(steps=forecast_steps).predicted_mean

            sarima_residuals_train = train_data[value_col] - sarima_fitted.fittedvalues
            sarima_residuals_train = sarima_residuals_train.dropna()

            if sarima_residuals_train.empty:
                raise ValueError("SARIMA residuals are empty after dropping NaNs. Check SARIMA parameters or data.")

            if sarima_msg_box is not None and sarima_msg_box.isVisible():
                sarima_msg_box.close()
            QApplication.processEvents()

            # --- Шаг 2: Обучаем LSTM работе с остатками SARIMA ---
            progress_dialog = QProgressDialog(f"Шаг 2/3: Обучение LSTM на остатках ({lstm_params['epochs']} эпох)...", "Отмена", 0, lstm_params['epochs'], self)
            progress_dialog.setWindowTitle("SARIMA-LSTM")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setWindowIcon(QIcon())
            progress_dialog.setWindowFlags(progress_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            progress_dialog.resize(250, 150)
            progress_dialog.show()
            QApplication.processEvents()

            residuals_df_train = pd.DataFrame(sarima_residuals_train)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_residuals_train = scaler.fit_transform(residuals_df_train.values)

            sequence_len = lstm_params['sequence_len']
            if sequence_len <= 0:
                raise ValueError("Длина последовательности LSTM (sequence_len) должна быть > 0.")

            X_residuals_train, y_residuals_train = [], []
            if len(scaled_residuals_train) <= sequence_len:
                raise ValueError(f"Insufficient residuals ({len(scaled_residuals_train)}) to create LSTM sequence of length {sequence_len}.")

            for i in range(sequence_len, len(scaled_residuals_train)):
                X_residuals_train.append(scaled_residuals_train[i-sequence_len:i, 0])
                y_residuals_train.append(scaled_residuals_train[i, 0])
            X_residuals_train, y_residuals_train = np.array(X_residuals_train), np.array(y_residuals_train)
            X_residuals_train = np.reshape(X_residuals_train, (X_residuals_train.shape[0], X_residuals_train.shape[1], 1))

            lstm_model_on_residuals = Sequential()
            lstm_model_on_residuals.add(LSTM(units=lstm_params['lstm_units'], input_shape=(X_residuals_train.shape[1], 1)))
            lstm_model_on_residuals.add(Dense(1))
            lstm_model_on_residuals.compile(optimizer='adam', loss='mean_squared_error')

            class ProgressCallback(Callback):
                def __init__(self, progress_dialog, total_epochs):
                    super().__init__()
                    self.progress_dialog = progress_dialog
                    self.total_epochs = total_epochs

                def on_epoch_end(self, epoch, logs=None):
                    self.progress_dialog.setValue(epoch + 1)
                    QApplication.processEvents()
                    if self.progress_dialog.wasCanceled():
                        self.model.stop_training = True

            history = None
            try:
                history = lstm_model_on_residuals.fit(
                    X_residuals_train, y_residuals_train,
                    epochs=lstm_params['epochs'],
                    batch_size=lstm_params['batch_size'],
                    verbose=0,
                    callbacks=[ProgressCallback(progress_dialog, lstm_params['epochs'])]
                )
            finally:
                if progress_dialog is not None:
                    progress_dialog.close()
                QApplication.processEvents()
                if history is None:
                    raise Exception("LSTM training on residuals was not completed.")

            # --- Шаг 3: Прогноз ---
            forecast_progress_dialog = QProgressDialog(f"Шаг 3/3: Прогнозирование остатков ({forecast_steps} шагов)...", "Отмена", 0, forecast_steps, self)
            forecast_progress_dialog.setWindowTitle("SARIMA-LSTM")
            forecast_progress_dialog.setWindowModality(Qt.WindowModal)
            forecast_progress_dialog.setMinimumDuration(0)
            forecast_progress_dialog.setWindowIcon(QIcon())
            forecast_progress_dialog.setWindowFlags(forecast_progress_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            forecast_progress_dialog.resize(250, 150)
            forecast_progress_dialog.show()
            QApplication.processEvents()

            last_sequence_scaled = scaled_residuals_train[-sequence_len:]
            current_batch = last_sequence_scaled.reshape((1, sequence_len, 1))

            lstm_residuals_forecast_scaled = []
            for i in range(forecast_steps):
                if forecast_progress_dialog.wasCanceled():
                    raise Exception("LSTM residual forecasting was cancelled.")
                current_pred_scaled = lstm_model_on_residuals.predict(current_batch, verbose=0)[0]
                lstm_residuals_forecast_scaled.append(current_pred_scaled)
                current_batch = np.append(current_batch[:, 1:, :], [[current_pred_scaled]], axis=1)
                forecast_progress_dialog.setValue(i + 1)
                QApplication.processEvents()

            lstm_residuals_forecast = scaler.inverse_transform(np.array(lstm_residuals_forecast_scaled).reshape(-1, 1)).flatten()

            if forecast_progress_dialog is not None:
                forecast_progress_dialog.close()
            QApplication.processEvents()

            # --- Шаг 4: комбинирование прогноза и данных и отображение ---
            sarima_forecast.index = test_data.index
            lstm_residuals_forecast_series = pd.Series(lstm_residuals_forecast, index=test_data.index)

            hybrid_forecast = sarima_forecast + lstm_residuals_forecast_series

            forecast_df = pd.DataFrame({'forecast': hybrid_forecast}, index=test_data.index)

            train_residuals_pred_scaled = lstm_model_on_residuals.predict(X_residuals_train, verbose=0)
            train_residuals_pred = scaler.inverse_transform(train_residuals_pred_scaled).flatten()
            fitted_residuals_index = sarima_residuals_train.index[sequence_len:]

            corresponding_sarima_fitted = sarima_fitted.fittedvalues.loc[fitted_residuals_index]

            hybrid_fitted_values = corresponding_sarima_fitted + train_residuals_pred
            hybrid_fitted_df = pd.DataFrame({'fitted': hybrid_fitted_values}, index=fitted_residuals_index)

            plot_forecast_df = pd.concat([hybrid_fitted_df, forecast_df])

            self.plot_data(self.current_df, forecast_df=plot_forecast_df,
                           title_suffix=" (Гибрид SARIMA-LSTM)",
                           split_point=self.train_df.index[-1])

            y_true = test_data[value_col]
            y_pred = forecast_df['forecast']
            self.calculate_and_display_metrics(y_true, y_pred, "SARIMA-LSTM")

        except Exception as e:
            if sarima_msg_box is not None and sarima_msg_box.isVisible():
                sarima_msg_box.close()
            if progress_dialog is not None:
                progress_dialog.close()
            if forecast_progress_dialog is not None:
                forecast_progress_dialog.close()
            QMessageBox.critical(self, "Ошибка SARIMA-LSTM", f"Произошла ошибка: {str(e)}\n{traceback.format_exc()}")
            print("---- SARIMA-LSTM Error Traceback ----")
            traceback.print_exc()
            print("------------------------------------")
        finally:
            if sarima_msg_box is not None and sarima_msg_box.isVisible():
                sarima_msg_box.close()
            if progress_dialog is not None:
                progress_dialog.close()
            if forecast_progress_dialog is not None:
                forecast_progress_dialog.close()
            QApplication.processEvents()



    def prophet_prediction(self):
        """Выполняем прогнозирование с помощью Facebook Prophet.""" 

        if self.current_df is None or self.value_col is None or self.date_col is None: 
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите и обработайте данные.") 
            return 

        prophet_msg_box = None
        try:

            data_to_fit_orig = self.train_df if self.train_df is not None else self.current_df 
            if data_to_fit_orig is None or data_to_fit_orig.empty:
                 QMessageBox.warning(self,"Нет данных для обучения", "Обучающая выборка пуста.") 
                 return 

            # --- 1. Подготовка данные для Prophet (требуются столбцы 'ds' и 'y') ---
            prophet_df_train = data_to_fit_orig.reset_index()
            # Получить название столбца даты
            date_col_name_in_df = prophet_df_train.columns[0] 
            # Переименуем столбцы в "ds" (метка даты) и "y" (значение).
            prophet_df_train = prophet_df_train.rename(
                columns={date_col_name_in_df: 'ds', self.value_col: 'y'}
            )
            prophet_df_train = prophet_df_train[['ds', 'y']] 
            # Проверка, что "ds" - это тип даты и времени
            prophet_df_train['ds'] = pd.to_datetime(prophet_df_train['ds'])

            # --- 2. Train Prophet model ---
            prophet_msg_box = QMessageBox(self)
            prophet_msg_box.setIcon(QMessageBox.Icon.Information) 
            prophet_msg_box.setText("Обучение модели Prophet...") 
            prophet_msg_box.setWindowTitle("Prophet") 
            prophet_msg_box.setStandardButtons(QMessageBox.StandardButton.Cancel)
            prophet_msg_box.show()
            QApplication.processEvents() 

            # Инициализировать модель Prophet
            model = FbProphet() 
            model.fit(prophet_df_train)


            if prophet_msg_box is not None and prophet_msg_box.isVisible():
                prophet_msg_box.close()
            QApplication.processEvents()

            # --- 3. Создаем фрейм данных "будущее" для прогнозирования --- 
            if self.test_df is not None:
                forecast_steps = len(self.test_df) # 
            else:
                forecast_steps = min(30, max(1, int(len(data_to_fit_orig) * 0.1))) 


            future_df = model.make_future_dataframe(periods=forecast_steps, freq=self.frequency)

            # --- 4. получаем предсказания ---
            forecast = model.predict(future_df)



            plot_forecast_df = forecast[['ds', 'yhat']].copy() # 
            plot_forecast_df = plot_forecast_df.rename(columns={'yhat': 'forecast'}) 
            plot_forecast_df = plot_forecast_df.set_index('ds')
            fitted_values_df = plot_forecast_df.iloc[:len(data_to_fit_orig)].rename(columns={'forecast': 'fitted'}) 
            forecast_values_df = plot_forecast_df.iloc[len(data_to_fit_orig):]


            final_plot_df = pd.concat([
                 fitted_values_df[['fitted']], 
                 forecast_values_df[['forecast']]
            ], sort=False) 

            # --- 6. Отображение результатов --- 
            self.plot_data(
                self.current_df, 
                forecast_df=final_plot_df, 
                title_suffix=" (Prophet)", 
                split_point=self.train_df.index[-1] if self.train_df is not None else None 
            )

            # Метрики
            if self.test_df is not None and not self.test_df.empty:
                 y_true = self.test_df[self.value_col]
                 if y_true.index.isin(final_plot_df.index).all():
                      y_pred = final_plot_df.loc[y_true.index, 'forecast']
                      self.calculate_and_display_metrics(y_true, y_pred, "Prophet") 
                 else:
                      print("Warning (Prophet): Not all test set indices found in forecast. Metrics not calculated.") 
                      QMessageBox.warning(self, "Ошибка метрик (Prophet)", "Не удалось сопоставить прогноз с тестовой выборкой для расчета метрик.") 

            #QMessageBox.information(self, "Prophet Завершено", "Прогнозирование с помощью Prophet завершено.") # "Prophet Complete", "Forecasting using Prophet finished."

        except Exception as e:

            if prophet_msg_box is not None and prophet_msg_box.isVisible():
                prophet_msg_box.close()
            QMessageBox.critical(self, "Ошибка Prophet", f"Ошибка при прогнозировании Prophet: {str(e)}\n{traceback.format_exc()}") 
        finally:
             if prophet_msg_box is not None and prophet_msg_box.isVisible(): 
                prophet_msg_box.close()
             QApplication.processEvents()



    def show_xgboost_config_dialog(self):
        """Показывает диалоговое окно настройки XGBoost и запускает прогнозирование."""

        if self.current_df is None or self.value_col is None: # 
             QMessageBox.warning(self, "Нет данных", "Сначала загрузите и обработайте данные.") 
             return
        # Recommend splitting data for XGBoost evaluation
        if self.train_df is None or self.test_df is None:
            QMessageBox.warning(self, "Разделение необходимо", "Для обучения и оценки XGBoost рекомендуется разделить данные на обучающую и тестовую выборки.") 

        dialog = XGBoostConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_params()
            self.xgboost_prediction(params) 

    def create_xgboost_features(self, df, target_col, n_lags=1):
        """Создает запаздывающие функции для модели XGBoost."""
        df_feat = df.copy() 
        for lag in range(1, n_lags + 1):
            df_feat[f'lag_{lag}'] = df_feat[target_col].shift(lag)
        df_feat = df_feat.dropna()# убираем пропуски
        return df_feat

    def xgboost_prediction(self, params):
        """Выполняет прогнозирование с помощью XGBoost.""" 
        if self.current_df is None or self.value_col is None: 
            QMessageBox.warning(self, "Нет данных", "Нет данных для обучения/тестирования XGBoost.")
            return

        xgb_msg_box = None
        try:
             value_col = self.value_col 
             n_lags = params['n_lags']

             # --- 1.Подготовка данных ---
             # создаем признаки
             df_with_features = self.create_xgboost_features(self.current_df[[value_col]], value_col, n_lags) 


             if df_with_features.empty: 
                raise ValueError(f"Недостаточно данных для создания {n_lags} лагов.") 

             target = df_with_features[value_col]
             features = df_with_features.drop(columns=[value_col])


             if self.train_df is not None and self.test_df is not None: 
                 split_date = self.train_df.index.intersection(features.index).max()
                 if pd.isna(split_date): 
                      QMessageBox.warning(self, "Предупреждение XGBoost", "Не удалось найти точку разделения после создания лагов. Обучение на всех данных.") 
                      X_train, y_train = features, target 
                      X_test, y_test = pd.DataFrame(), pd.Series() 
                 else:

                      train_mask = features.index <= split_date 
                      test_mask = features.index > split_date
                      X_train, y_train = features[train_mask], target[train_mask]
                      X_test, y_test = features[test_mask], target[test_mask] 
             else:

                 X_train, y_train = features, target 
                 X_test, y_test = pd.DataFrame(), pd.Series()
                 split_date = None 


             if X_train.empty or y_train.empty: 
                 raise ValueError("Обучающая выборка пуста после создания признаков.") 
             # --- 2. Обучаем XGBoost модель--- 
             xgb_msg_box = QMessageBox(self)
             xgb_msg_box.setIcon(QMessageBox.Icon.Information) 
             xgb_msg_box.setText("Обучение модели XGBoost...") 
             xgb_msg_box.setWindowTitle("XGBoost")
             xgb_msg_box.setStandardButtons(QMessageBox.StandardButton.Cancel) 
             xgb_msg_box.show()
             QApplication.processEvents() 


             model = xgb.XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'], 
                colsample_bytree=params['colsample_bytree'], 
                objective='reg:squarederror', 
                random_state=42 
             )


             model.fit(X_train, y_train)


             if xgb_msg_box is not None and xgb_msg_box.isVisible(): 
                xgb_msg_box.close()
             QApplication.processEvents()

             # --- 3. Прогноз ---
             fitted_values = model.predict(X_train) # 
             fitted_df = pd.DataFrame({'fitted': fitted_values}, index=X_train.index)

             forecast_df = pd.DataFrame(index=X_test.index, columns=['forecast']) # Initialize with test index 
             if not X_test.empty:
                predictions = model.predict(X_test)
                forecast_df['forecast'] = predictions 

                if not forecast_df.empty and 'forecast' in forecast_df.columns:
                    # 1. Сместить индекс прогноза на один временной шаг назад
                    original_index = forecast_df.index
                    shifted_index = None

                    if isinstance(original_index, pd.DatetimeIndex):
                        freq_offset = None
                        # Попытаться использовать собственную частоту индекса
                        if original_index.freq:
                            freq_offset = original_index.freq
                        # Если нет, использовать частоту, указанную пользователем в интерфейсе
                        elif self.frequency:
                            try:
                                freq_offset = pd.tseries.frequencies.to_offset(self.frequency)
                            except ValueError as e:
                                print(f"Предупреждение (XGBoost): Не удалось преобразовать self.frequency '{self.frequency}' в смещение. Ошибка: {e}")
                        
                        if freq_offset:
                            try:
                                shifted_index = original_index - freq_offset
                            except Exception as e: # Общее исключение для проблем с вычитанием
                                print(f"Предупреждение (XGBoost): Не удалось вычесть смещение частоты для сдвига индекса. Ошибка: {e}")
                        else:
                            print("Предупреждение (XGBoost): Отсутствует информация о частоте для сдвига DatetimeIndex.")
                    elif isinstance(original_index, (pd.RangeIndex, pd.Int64Index)): # Если индекс числовой
                        shifted_index = original_index - 1
                    else:
                        print(f"Предупреждение (XGBoost): Тип индекса '{type(original_index)}' не обработан для сдвига.")

                    if shifted_index is not None:
                        forecast_df.index = shifted_index
                    else:
                        print("Предупреждение (XGBoost): Индекс прогноза не был сдвинут. Будет удалено первое значение из исходного (несдвинутого) прогноза.")

                    # 2. Удалить первое значение (потенциально сдвинутого) прогноза
                    if not forecast_df.empty: # Проверить еще раз, на случай если DataFrame был маленьким
                        forecast_df = forecast_df.iloc[1:]


             # --- 4. Итоги и отображение ---

             plot_forecast_df = pd.concat([fitted_df, forecast_df]) # 
             plot_split_point = split_date 

             # Грфик
             self.plot_data(self.current_df, 
                           forecast_df=plot_forecast_df,
                           title_suffix=" (XGBoost)", 
                           split_point=plot_split_point) 

           
             if not y_test.empty:
                y_true = y_test 
                y_pred = forecast_df['forecast'].dropna()
                self.calculate_and_display_metrics(y_true, y_pred, "XGBoost")
             else:
                 QMessageBox.information(self,"XGBoost","Обучение XGBoost завершено (без тестовой выборки).") 

        except Exception as e: 

            if xgb_msg_box is not None and xgb_msg_box.isVisible(): 
                xgb_msg_box.close()
            QMessageBox.critical(self, "Ошибка XGBoost", f"Ошибка при прогнозировании XGBoost: {str(e)}\n{traceback.format_exc()}") 
        finally:
             if xgb_msg_box is not None and xgb_msg_box.isVisible():
                 xgb_msg_box.close()
             QApplication.processEvents()


    def calculate_and_display_metrics(self, y_true, y_pred, model_name):
        """Вычисляет MAE, MSE, RMSE, MAPE и отображает их в QMessageBox."""
        if y_true is None or y_pred is None: 
            print(f"Warning: Cannot calculate metrics for {model_name}. Missing y_true or y_pred.") 
            return
        if not isinstance(y_true, pd.Series):
             y_true = pd.Series(y_true)
        if not isinstance(y_pred, pd.Series): 
             try:
                 y_pred = pd.Series(y_pred, index=y_true.index[:len(y_pred)]) 
             except Exception as series_err:
                 print(f"Warning: Could not create pd.Series for y_pred ({model_name}): {series_err}") 
                 return

        try:
            common_index = y_true.dropna().index.intersection(y_pred.dropna().index)

            if common_index.empty: 
                 QMessageBox.warning(self, f"Ошибка расчета метрик ({model_name})", 
                                       "Нет совпадающих не-пропущенных значений между фактическими и предсказанными данными.")
                 return 

            y_true_aligned = y_true.loc[common_index] 
            y_pred_aligned = y_pred.loc[common_index]


            if len(y_true_aligned) == 0:
                  QMessageBox.warning(self, f"Ошибка расчета метрик ({model_name})", 
                                       "Нет валидных данных для расчета метрик после выравнивания.") 
                  return


            mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
            mse = mean_squared_error(y_true_aligned, y_pred_aligned)
            rmse = np.sqrt(mse)


            mask = y_true_aligned != 0 
            if np.any(mask): 
                 mape = mean_absolute_percentage_error(y_true_aligned[mask], y_pred_aligned[mask]) * 100 #
                 mape_str = f"{mape:.4f}%"

                 if not np.all(mask):
                     mape_str += " (рассчитано без нулевых значений)" 
            else:
                 mape_str = "N/A (все фактические значения 0)" 


            metrics_text = (
                f"Метрики для модели: {model_name}\n" 
                 f"(на {len(y_true_aligned)} точках тестовой выборки)\n\n" 
                f"MAE:  {mae:.4f}\n"
                f"MSE:  {mse:.4f}\n" 
                f"RMSE: {rmse:.4f}\n"
                f"MAPE: {mape_str}" 
            )

            QMessageBox.information(self, f"Метрики: {model_name}", metrics_text) 

        except Exception as e:
            QMessageBox.critical(self, f"Ошибка расчета метрик ({model_name})", f"Не удалось рассчитать метрики: {str(e)}\n{traceback.format_exc()}") 


if __name__ == '__main__':
    # Стандартная настройка приложения Qt
    app = QApplication(sys.argv)
    # Примените стиль фьюжн для лучшего кроссплатформенного оформления
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())