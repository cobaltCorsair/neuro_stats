#!/usr/bin/env python3
"""
Скрипт автоматической сборки RadiobiologyAnalysis.exe
Кроссплатформенная версия на Python
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil
from datetime import datetime

# Цвета для терминала (ANSI)
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GRAY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header():
    """Выводит заголовок"""
    print()
    print(f"{Colors.CYAN}{'='*50}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}  Radiobiology Analysis - Build Script{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*50}{Colors.RESET}")
    print()

def check_poetry():
    """Проверяет наличие Poetry"""
    try:
        result = subprocess.run(['poetry', '--version'],
                              capture_output=True,
                              text=True,
                              check=True)
        print(f"{Colors.GREEN}[✓] Poetry found: {result.stdout.strip()}{Colors.RESET}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Colors.RED}[✗] Poetry not found!{Colors.RESET}")
        print(f"{Colors.YELLOW}Please install Poetry: https://python-poetry.org/docs/#installation{Colors.RESET}")
        return False

def show_env_info():
    """Показывает информацию об окружении"""
    print()
    print(f"{Colors.YELLOW}[1/4] Checking Poetry environment...{Colors.RESET}")
    subprocess.run(['poetry', 'env', 'info'])

def clean_previous_build():
    """Очищает предыдущую сборку"""
    print()
    print(f"{Colors.YELLOW}[2/4] Cleaning previous build...{Colors.RESET}")

    exe_path = Path('dist/RadiobiologyAnalysis.exe')
    if exe_path.exists():
        exe_path.unlink()
        print(f"{Colors.GRAY}Previous build deleted.{Colors.RESET}")
    else:
        print(f"{Colors.GRAY}No previous build found.{Colors.RESET}")

def build_application():
    """Собирает приложение"""
    print()
    print(f"{Colors.YELLOW}[3/4] Building application with PyInstaller...{Colors.RESET}")
    print(f"{Colors.GRAY}This may take several minutes...{Colors.RESET}")
    print()

    start_time = datetime.now()

    try:
        subprocess.run(
            ['poetry', 'run', 'pyinstaller', 'radiobiology_app.spec', '--clean'],
            check=True
        )

        end_time = datetime.now()
        build_time = (end_time - start_time).total_seconds()

        return True, build_time
    except subprocess.CalledProcessError:
        return False, 0

def check_result(build_time):
    """Проверяет результат сборки"""
    print()
    print(f"{Colors.YELLOW}[4/4] Checking build result...{Colors.RESET}")

    exe_path = Path('dist/RadiobiologyAnalysis.exe')

    if exe_path.exists():
        file_size_mb = exe_path.stat().st_size / (1024 * 1024)

        print()
        print(f"{Colors.GREEN}{'='*50}{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}  BUILD SUCCESSFUL!{Colors.RESET}")
        print(f"{Colors.GREEN}{'='*50}{Colors.RESET}")
        print()
        print(f"  File: {exe_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Build time: {build_time:.1f} seconds")
        print()

        return True
    else:
        print()
        print(f"{Colors.RED}{'='*50}{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}  BUILD FAILED!{Colors.RESET}")
        print(f"{Colors.RED}{'='*50}{Colors.RESET}")
        print(f"{Colors.YELLOW}Check the errors above.{Colors.RESET}")
        print()
        return False

def ask_run():
    """Спрашивает, запустить ли приложение"""
    try:
        response = input("Do you want to run the application now? (Y/N): ").strip().lower()
        if response in ['y', 'yes']:
            print()
            print(f"{Colors.CYAN}Launching application...{Colors.RESET}")

            if sys.platform == 'win32':
                os.startfile('dist/RadiobiologyAnalysis.exe')
            else:
                subprocess.Popen(['./dist/RadiobiologyAnalysis.exe'])
    except KeyboardInterrupt:
        print()

def main():
    """Главная функция"""
    # Переходим в директорию скрипта
    os.chdir(Path(__file__).parent)

    print_header()

    # Проверка Poetry
    if not check_poetry():
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Информация об окружении
    show_env_info()

    # Очистка
    clean_previous_build()

    # Сборка
    success, build_time = build_application()

    if not success:
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Проверка результата
    if check_result(build_time):
        ask_run()
    else:
        input("\nPress Enter to exit...")
        sys.exit(1)

    print()
    print(f"{Colors.CYAN}Build script completed.{Colors.RESET}")
    print()
    input("Press Enter to exit...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Build cancelled by user.{Colors.RESET}")
        sys.exit(1)
