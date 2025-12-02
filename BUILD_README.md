# 🏗️ Инструкция по сборке Radiobiology Analysis

## 🚀 Быстрая сборка

### Автоматические скрипты сборки

В проекте есть 3 скрипта для автоматической сборки:

#### 1. **build.bat** - для Windows Command Prompt
```cmd
build.bat
```

#### 2. **build.ps1** - для Windows PowerShell
```powershell
.\build.ps1
```

#### 3. **build.py** - кроссплатформенный Python скрипт
```bash
python build.py
```

### Что делают скрипты:

✅ Проверяют наличие и версию Poetry
✅ Показывают информацию об окружении
✅ Очищают предыдущую сборку
✅ Запускают PyInstaller через Poetry
✅ Показывают размер и время сборки
✅ Предлагают запустить приложение

---

## 📋 Ручная сборка

Если нужно собрать вручную:

```bash
cd c:\dev\neuro_stats
poetry run pyinstaller radiobiology_app.spec --clean
```

Результат: `dist/RadiobiologyAnalysis.exe`

---

## 📦 Результат сборки

| Параметр | Значение |
|----------|----------|
| **Финальный размер** | **~116 MB** |
| **Точка входа** | `work_with_prepared_data/radiobioligy_project/gui/main_window.py` |
| **Конфигурация** | `radiobiology_app.spec` |
| **Python версия** | 3.10+ (Poetry окружение) |

---

## 🔧 Технические детали

### Зависимости (из Poetry окружения)

**Основные:**
- Python: >=3.10,<3.13
- PyQt6: ^6.6.1
- numpy: 1.26.4
- scipy: 1.11.4 (важна версия!)
- matplotlib: ^3.7.0
- pandas: ^2.0.0
- seaborn: ^0.13.2
- openpyxl: ^3.1.2
- fastdtw: ^0.3.4
- scikit-learn: ^1.3.0

**Dev:**
- pyinstaller: ^6.0.0

### Оптимизации в сборке

✂️ **Исключены тяжелые модули:**
- PyQt6: WebEngine, QML, Network, Multimedia
- matplotlib: backends для GTK, Tk, macOS
- Все тесты библиотек
- Документация и примеры

🔬 **SciPy:**
- Версия 1.11.4 (стабильная для PyInstaller)
- Включены все модули (из-за взаимных зависимостей)
- Исключены только тесты

💾 **Data files:**
- Отключен сбор data files для уменьшения размера

---

## ⚠️ Важные примечания

### Версия scipy
**НЕ обновляйте scipy выше 1.11.4!**
Версии 1.13+ и 1.16+ имеют проблемы совместимости с PyInstaller.

### Окружение Poetry
Сборка **обязательно** должна выполняться из Poetry окружения:
```bash
poetry env info  # проверить окружение
```

### Первый запуск
При первом запуске exe может быть медленнее - PyInstaller распаковывает файлы.

---

## 🐛 Troubleshooting

### "Poetry not found"
Установите Poetry:
```bash
pip install poetry
```

### "ModuleNotFoundError"
Убедитесь, что все зависимости установлены:
```bash
poetry install
```

### Сборка падает с ошибкой
1. Очистите кеш PyInstaller:
   ```bash
   rm -rf build/ dist/
   ```
2. Пересоберите с флагом `--clean`

### Приложение не запускается
Запустите из командной строки, чтобы увидеть ошибки:
```bash
cd dist
RadiobiologyAnalysis.exe
```

---

## 📁 Структура файлов

```
neuro_stats/
├── build.bat                    # Скрипт сборки (Windows CMD)
├── build.ps1                    # Скрипт сборки (PowerShell)
├── build.py                     # Скрипт сборки (Python)
├── radiobiology_app.spec        # Конфигурация PyInstaller
├── pyi_rth_scipy.py            # Runtime hook для scipy
├── pyproject.toml              # Зависимости Poetry
├── dist/                       # Результат сборки
│   └── RadiobiologyAnalysis.exe
└── work_with_prepared_data/
    └── radiobioligy_project/
        └── gui/
            └── main_window.py   # Точка входа
```

---

## 📈 История оптимизации

| Этап | Размер | Комментарий |
|------|--------|-------------|
| Исходная | 165 MB | С глобальными пакетами |
| Оптимизированная | **116 MB** | **↓49 MB (30%)** |

**Оптимизации:**
- Сборка из чистого Poetry окружения
- Исключение ненужных модулей PyQt6
- Исключение тестов
- Отключение data files

---

## 📞 Поддержка

При проблемах со сборкой проверьте:
1. Версию Python (должна быть 3.10-3.12)
2. Версию scipy (должна быть 1.11.4)
3. Что используется окружение Poetry
4. Логи сборки на наличие ошибок
