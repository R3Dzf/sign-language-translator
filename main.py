"""
GUI launcher for the Sign Language Translator project.
"""

from __future__ import annotations

import sys

from src.gui_app import main as gui_main
from src.runtime_setup import suppress_runtime_warnings

suppress_runtime_warnings()


def main() -> None:
    if "--cli" in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != "--cli"]
        from src.collect_data import main as collect_data_main
        from src.collect_sequence_data import main as collect_sequence_data_main
        from src.real_time_dynamic_translate import run_dynamic_translator
        from src.real_time_translate import main as translator_main
        from src.train_sequence_model import main as train_sequence_model_main
        from src.train_model import main as train_model_main

        while True:
            print("\nSign Language Translator")
            print("========================")
            print("1. Collect static data")
            print("2. Train static model")
            print("3. Run static translator")
            print("4. Collect dynamic sequence data")
            print("5. Train dynamic model")
            print("6. Run dynamic translator")
            print("7. Run hybrid translator")
            print("8. Exit")
            choice = input("Choose an option: ").strip()

            if choice == "1":
                collect_data_main()
            elif choice == "2":
                train_model_main()
            elif choice == "3":
                translator_main()
            elif choice == "4":
                collect_sequence_data_main()
            elif choice == "5":
                train_sequence_model_main()
            elif choice == "6":
                run_dynamic_translator("dynamic")
            elif choice == "7":
                run_dynamic_translator("hybrid")
            elif choice == "8":
                print("Exiting.")
                break
            else:
                print("Invalid choice. Please enter a number from 1 to 8.")
    else:
        gui_main()


if __name__ == "__main__":
    main()
