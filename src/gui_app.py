"""
Tkinter GUI for the Sign Language Translator project.
"""

from __future__ import annotations

import io
import queue
import threading
import tkinter as tk
from contextlib import redirect_stdout
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from src.collect_data import collect_samples_for_class
from src.collect_sequence_data import collect_sequence_data
from src.config import (
    ASSETS_DIR,
    BUNDLED_ASSETS_DIR,
    CLASS_LABELS,
    DEFAULT_SAMPLES_PER_CLASS,
    DYNAMIC_SIGN_LABELS,
    RAW_DATASET_DIR,
    SEQUENCE_DATASET_DIR,
)
from src.real_time_dynamic_translate import DynamicTranslatorSession, HybridTranslatorSession
from src.real_time_translate import TranslatorSession, draw_translator_overlay
from src.runtime_setup import suppress_runtime_warnings
from src.train_model import train_model
from src.train_sequence_model import train_sequence_model


suppress_runtime_warnings()


class SignLanguageGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ASL Sign Language Translator")
        self.root.geometry("1280x860")
        self.root.configure(bg="#edf4f7")
        self.assets_dir = ASSETS_DIR if ASSETS_DIR.exists() else BUNDLED_ASSETS_DIR
        self.app_icon_png = self.assets_dir / "app_icon_source.png"
        self.app_icon_ico = self.assets_dir / "app_icon.ico"
        self.app_icon_photo = None
        self._configure_app_icon()

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.layout(
            "TNotebook.Tab",
            [
                (
                    "Notebook.tab",
                    {
                        "sticky": "nswe",
                        "children": [
                            (
                                "Notebook.padding",
                                {
                                    "side": "top",
                                    "sticky": "nswe",
                                    "children": [
                                        (
                                            "Notebook.label",
                                            {
                                                "side": "top",
                                                "sticky": "",
                                            },
                                        )
                                    ],
                                },
                            )
                        ],
                    },
                )
            ],
        )
        self.style.configure(
            "TNotebook",
            background="#edf4f7",
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        self.style.configure(
            "TNotebook.Tab",
            padding=(24, 12),
            width=16,
            font=("Segoe UI", 11, "bold"),
            background="#d9cdbf",
            foreground="#123047",
            borderwidth=0,
            relief="flat",
        )
        self.style.map(
            "TNotebook.Tab",
            background=[
                ("selected", "#0f766e"),
                ("active", "#d9cdbf"),
                ("!selected", "#d9cdbf"),
            ],
            foreground=[
                ("selected", "white"),
                ("active", "#123047"),
                ("!selected", "#123047"),
            ],
            padding=[
                ("selected", (24, 12)),
                ("active", (24, 12)),
                ("!selected", (24, 12)),
            ],
            lightcolor=[
                ("selected", "#0f766e"),
                ("active", "#d9cdbf"),
                ("!selected", "#d9cdbf"),
            ],
            darkcolor=[
                ("selected", "#0f766e"),
                ("active", "#d9cdbf"),
                ("!selected", "#d9cdbf"),
            ],
            bordercolor=[
                ("selected", "#0f766e"),
                ("active", "#d9cdbf"),
                ("!selected", "#d9cdbf"),
            ],
            focuscolor=[
                ("selected", "#0f766e"),
                ("active", "#d9cdbf"),
                ("!selected", "#d9cdbf"),
            ],
        )
        self.style.configure("TFrame", background="#edf4f7")
        self.style.configure("Card.TFrame", background="#ffffff")
        self.style.configure("Title.TLabel", background="#edf4f7", foreground="#123047", font=("Segoe UI", 24, "bold"))
        self.style.configure("Body.TLabel", background="#edf4f7", foreground="#36566f", font=("Segoe UI", 11))
        self.style.configure("CardTitle.TLabel", background="#ffffff", foreground="#123047", font=("Segoe UI", 15, "bold"))
        self.style.configure("CardBody.TLabel", background="#ffffff", foreground="#4a6277", font=("Segoe UI", 10))
        self.style.configure("AboutTitle.TLabel", background="#ffffff", foreground="#123047", font=("Segoe UI", 22, "bold"))
        self.style.configure("AboutName.TLabel", background="#ffffff", foreground="#18415d", font=("Segoe UI", 14, "bold"))
        self.style.configure("AboutMeta.TLabel", background="#ffffff", foreground="#49657a", font=("Segoe UI", 13))
        self.style.configure("AboutAccent.TLabel", background="#ffffff", foreground="#0f766e", font=("Segoe UI", 16, "bold"))
        self.style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=10)

        self.translator_session = None
        self.translator_running = False
        self.training_thread = None
        self.collect_thread = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.video_placeholder_text = "Camera preview will appear here"

        self._build_layout()
        self.collection_mode_var.trace_add("write", self._on_collection_mode_change)
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll_logs()

    def _configure_app_icon(self) -> None:
        if self.app_icon_ico.exists():
            try:
                self.root.iconbitmap(default=str(self.app_icon_ico))
            except Exception:
                pass

        if self.app_icon_png.exists():
            try:
                self.app_icon_photo = tk.PhotoImage(file=str(self.app_icon_png))
                self.root.iconphoto(True, self.app_icon_photo)
            except Exception:
                self.app_icon_photo = None

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=28, pady=(20, 10))
        ttk.Label(header, text="ASL Sign Language Translator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Faster camera startup, calmer prediction logic, and a cleaner workflow for data collection, training, and live translation.",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        self.home_tab = ttk.Frame(notebook)
        self.collect_tab = ttk.Frame(notebook)
        self.train_tab = ttk.Frame(notebook)
        self.translate_tab = ttk.Frame(notebook)
        self.about_tab = ttk.Frame(notebook)

        notebook.add(self.translate_tab, text="Translator")
        notebook.add(self.collect_tab, text="Collect Data")
        notebook.add(self.train_tab, text="Train Model")
        notebook.add(self.home_tab, text="Dashboard")
        notebook.add(self.about_tab, text="About")

        self._build_home_tab()
        self._build_collect_tab()
        self._build_train_tab()
        self._build_translate_tab()
        self._build_about_tab_v4()

    def _build_home_tab(self) -> None:
        cards = ttk.Frame(self.home_tab)
        cards.pack(fill="both", expand=True, padx=12, pady=12)

        self._make_card(cards, "Dataset", "Collect landmark samples for each ASL sign with an OpenCV recording window.", 0, 0)
        self._make_card(cards, "Training", "Train the Random Forest model and save the model plus label encoder.", 0, 1)
        self._make_card(cards, "Translator", "Run the live translator with improved stabilization and automatic sentence building.", 1, 0)
        self._make_card(cards, "Quality", "Automatic writing now requires a stable sign and at least 80% confidence.", 1, 1)

        for index in range(2):
            cards.columnconfigure(index, weight=1)

    def _make_card(self, parent, title: str, text: str, row: int, column: int) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=22)
        card.grid(row=row, column=column, sticky="nsew", padx=12, pady=12)
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text=text, style="CardBody.TLabel", wraplength=360, justify="left").pack(anchor="w", pady=(10, 0))

    def _build_collect_tab(self) -> None:
        wrapper = ttk.Frame(self.collect_tab, padding=16)
        wrapper.pack(fill="both", expand=True)

        form = ttk.Frame(wrapper, style="Card.TFrame", padding=24)
        form.pack(fill="x")

        ttk.Label(form, text="Collect Custom ASL Samples", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(form, text="Collection Type", style="CardBody.TLabel").grid(row=1, column=0, sticky="w", pady=(18, 6))

        self.collection_mode_var = tk.StringVar(value="Static signs")
        self.class_var = tk.StringVar(value=CLASS_LABELS[0])
        self.dynamic_class_var = tk.StringVar(value=DYNAMIC_SIGN_LABELS[0])
        self.custom_class_var = tk.StringVar(value="")
        self.samples_var = tk.StringVar(value=str(DEFAULT_SAMPLES_PER_CLASS))
        self.save_dir_var = tk.StringVar(value=str(RAW_DATASET_DIR))
        self.collect_status_var = tk.StringVar(value="Ready to collect.")

        ttk.Combobox(
            form,
            textvariable=self.collection_mode_var,
            values=["Static signs", "Dynamic signs"],
            state="readonly",
            width=30,
        ).grid(row=1, column=1, sticky="ew", pady=(18, 6))
        ttk.Label(form, text="Static Label", style="CardBody.TLabel").grid(row=2, column=0, sticky="w", pady=(14, 6))
        ttk.Combobox(form, textvariable=self.class_var, values=CLASS_LABELS, state="readonly", width=30).grid(row=2, column=1, sticky="ew", pady=(14, 6))
        ttk.Label(form, text="Dynamic Label", style="CardBody.TLabel").grid(row=3, column=0, sticky="w", pady=(14, 6))
        ttk.Combobox(form, textvariable=self.dynamic_class_var, values=DYNAMIC_SIGN_LABELS, state="readonly", width=30).grid(row=3, column=1, sticky="ew", pady=(14, 6))
        ttk.Label(form, text="Or New Label", style="CardBody.TLabel").grid(row=4, column=0, sticky="w", pady=(14, 6))
        ttk.Entry(form, textvariable=self.custom_class_var, width=30).grid(row=4, column=1, sticky="ew", pady=(14, 6))
        ttk.Label(form, text="Samples / Sequences", style="CardBody.TLabel").grid(row=5, column=0, sticky="w", pady=(14, 6))
        ttk.Entry(form, textvariable=self.samples_var, width=20).grid(row=5, column=1, sticky="ew", pady=(14, 6))
        ttk.Label(form, text="Save Folder", style="CardBody.TLabel").grid(row=6, column=0, sticky="w", pady=(14, 6))
        ttk.Entry(form, textvariable=self.save_dir_var, width=50).grid(row=6, column=1, sticky="ew", pady=(14, 6))

        self.collect_button = tk.Button(
            form,
            text="Start Collection",
            command=self.start_collection,
            bg="#0f766e",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=18,
            pady=10,
        )
        self.collect_button.grid(row=7, column=0, columnspan=2, sticky="w", pady=(20, 0))
        ttk.Label(form, textvariable=self.collect_status_var, style="Body.TLabel").grid(row=8, column=0, columnspan=2, sticky="w", pady=(14, 0))
        form.columnconfigure(1, weight=1)
        self._on_collection_mode_change()

    def _on_collection_mode_change(self, *_args) -> None:
        mode_name = self.collection_mode_var.get().strip()
        current_path = self.save_dir_var.get().strip()

        if mode_name == "Dynamic signs":
            if not current_path or current_path == str(RAW_DATASET_DIR):
                self.save_dir_var.set(str(SEQUENCE_DATASET_DIR))
            self.collect_status_var.set("Dynamic collection will save sequences to dataset/sequences.")
        else:
            if not current_path or current_path == str(SEQUENCE_DATASET_DIR):
                self.save_dir_var.set(str(RAW_DATASET_DIR))
            self.collect_status_var.set("Static collection will save landmark rows to dataset/raw.")

    def _build_train_tab(self) -> None:
        wrapper = ttk.Frame(self.train_tab, padding=16)
        wrapper.pack(fill="both", expand=True)

        panel = ttk.Frame(wrapper, style="Card.TFrame", padding=24)
        panel.pack(fill="both", expand=True)

        ttk.Label(panel, text="Train Landmark Model", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(panel, text="This merges dataset files, trains the model, and saves the report.", style="CardBody.TLabel").pack(anchor="w", pady=(8, 16))
        self.training_mode_var = tk.StringVar(value="Static model")
        ttk.Label(panel, text="Training Type", style="CardBody.TLabel").pack(anchor="w", pady=(0, 6))
        ttk.Combobox(
            panel,
            textvariable=self.training_mode_var,
            values=["Static model", "Dynamic model"],
            state="readonly",
            width=24,
        ).pack(anchor="w", pady=(0, 14))

        self.train_button = tk.Button(
            panel,
            text="Train Model",
            command=self.start_training,
            bg="#1d4ed8",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=18,
            pady=10,
        )
        self.train_button.pack(anchor="w")

        self.log_text = tk.Text(panel, height=26, bg="#0f172a", fg="#dbeafe", insertbackground="white", font=("Consolas", 10), relief="flat")
        self.log_text.pack(fill="both", expand=True, pady=(18, 0))

    def _build_about_tab(self) -> None:
        wrapper = ttk.Frame(self.about_tab, padding=24)
        wrapper.pack(fill="both", expand=True)
        wrapper.columnconfigure(0, weight=1)
        wrapper.rowconfigure(0, weight=1)

        card = ttk.Frame(wrapper, style="Card.TFrame", padding=32)
        card.grid(row=0, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)

        ttk.Label(card, text="About The Team", style="AboutTitle.TLabel", anchor="center", justify="center").grid(row=0, column=0, pady=(0, 12))
        ttk.Label(
            card,
            text="Sign Language Translator Project",
            style="AboutAccent.TLabel",
            anchor="center",
            justify="center",
        ).grid(row=1, column=0, pady=(0, 24))

        team_names = [
            "أحمد يوسف يوسف منصور بوشه",
            "ابانوب امير جبران جرجس جبران",
            "محمود السيد على السعيد الترابى",
            "انس محمد سعيد محمد عفيفى",
        ]

        names_frame = ttk.Frame(card, style="Card.TFrame")
        names_frame.grid(row=2, column=0, sticky="ew", pady=(0, 28))
        names_frame.columnconfigure(0, weight=1)

        for index, name in enumerate(team_names):
            ttk.Label(
                names_frame,
                text=name,
                style="AboutName.TLabel",
                anchor="center",
                justify="center",
            ).grid(row=index, column=0, pady=8, sticky="ew")

        ttk.Label(
            card,
            text="قسم حاسبات وتحكم آلي",
            style="AboutAccent.TLabel",
            anchor="center",
            justify="center",
        ).grid(row=3, column=0, pady=(6, 10))
        ttk.Label(card, text="مستوى اول", style="AboutMeta.TLabel", anchor="center", justify="center").grid(row=4, column=0, pady=4)
        ttk.Label(
            card,
            text="كلية الهندسة - جامعة طنطا",
            style="AboutMeta.TLabel",
            anchor="center",
            justify="center",
        ).grid(row=5, column=0, pady=4)

    def _build_about_tab_v2(self) -> None:
        self.style.configure("AboutCard.TFrame", background="#f7f1ea")
        self.style.configure("AboutInner.TFrame", background="#fffdf9")
        self.style.configure("AboutTitle.TLabel", background="#f7f1ea", foreground="#123047", font=("Segoe UI", 22, "bold"))
        self.style.configure("AboutName.TLabel", background="#fffdf9", foreground="#18415d", font=("Segoe UI", 14, "bold"))
        self.style.configure("AboutMeta.TLabel", background="#f7f1ea", foreground="#49657a", font=("Segoe UI", 13))
        self.style.configure("AboutAccent.TLabel", background="#f7f1ea", foreground="#0f766e", font=("Segoe UI", 16, "bold"))
        self.style.configure("AboutBio.TLabel", background="#fffdf9", foreground="#3c5568", font=("Segoe UI", 12), wraplength=900, justify="center")

        for child in self.about_tab.winfo_children():
            child.destroy()

        wrapper = ttk.Frame(self.about_tab, padding=24)
        wrapper.pack(fill="both", expand=True)
        wrapper.columnconfigure(0, weight=1)
        wrapper.rowconfigure(0, weight=1)

        card = ttk.Frame(wrapper, style="AboutCard.TFrame", padding=32)
        card.grid(row=0, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)

        
        ttk.Label(card, text="Sign Language Translator Project", style="AboutAccent.TLabel", anchor="center", justify="center").grid(row=1, column=0, pady=(0, 24))

        bio_frame = ttk.Frame(card, style="AboutInner.TFrame", padding=22)
        bio_frame.grid(row=2, column=0, sticky="ew", pady=(0, 22))
        bio_frame.columnconfigure(0, weight=1)
        ttk.Label(card, text="إعداد الفريق", style="AboutTitle.TLabel", anchor="center", justify="center").grid(row=0, column=0, pady=(0, 12))
        ttk.Label(
            bio_frame,
            text="About the project: This project aims to build an instant translator for American Sign Language using a camera, computer vision, and machine learning technologies. It will recognize hand gestures and convert them into text in real time, with the potential for future expansion to add more gestures and improve the user experience.",
            style="AboutBio.TLabel",
            anchor="center",
            justify="center",
        ).grid(row=0, column=0, sticky="ew")

        team_names = [
            "أحمد يوسف يوسف منصور بوشه",
            "ابانوب امير جبران جرجس جبران",
            "محمود السيد على السعيد الترابى",
            "انس محمد سعيد محمد عفيفى",
        ]

        names_frame = ttk.Frame(card, style="AboutInner.TFrame", padding=18)
        names_frame.grid(row=3, column=0, sticky="ew", pady=(0, 28))
        names_frame.columnconfigure(0, weight=1)

        for index, name in enumerate(team_names):
            ttk.Label(names_frame, text=name, style="AboutName.TLabel", anchor="center", justify="center").grid(row=index * 2, column=0, pady=8, sticky="ew")
            if index < len(team_names) - 1:
                ttk.Separator(names_frame, orient="horizontal").grid(row=index * 2 + 1, column=0, sticky="ew", pady=4)

        ttk.Label(card, text="قسم حاسبات وتحكم آلي", style="AboutAccent.TLabel", anchor="center", justify="center").grid(row=4, column=0, pady=(6, 10))
        ttk.Label(card, text="مستوى اول", style="AboutMeta.TLabel", anchor="center", justify="center").grid(row=5, column=0, pady=4)
        ttk.Label(card, text="كلية الهندسة - جامعة طنطا", style="AboutMeta.TLabel", anchor="center", justify="center").grid(row=6, column=0, pady=4)

    def _build_about_tab_v3(self) -> None:
        self.style.configure("AboutCard.TFrame", background="#f7f1ea")
        self.style.configure("AboutInner.TFrame", background="#fffdf9")
        self.style.configure("AboutTitle.TLabel", background="#f7f1ea", foreground="#123047", font=("Segoe UI", 22, "bold"))
        self.style.configure("AboutName.TLabel", background="#fffdf9", foreground="#18415d", font=("Segoe UI", 14, "bold"))
        self.style.configure("AboutMeta.TLabel", background="#f7f1ea", foreground="#49657a", font=("Segoe UI", 13))
        self.style.configure("AboutAccent.TLabel", background="#f7f1ea", foreground="#0f766e", font=("Segoe UI", 16, "bold"))
        self.style.configure("AboutBio.TLabel", background="#fffdf9", foreground="#3c5568", font=("Segoe UI", 12), wraplength=900, justify="center")

        for child in self.about_tab.winfo_children():
            child.destroy()

        wrapper = ttk.Frame(self.about_tab, padding=18)
        wrapper.pack(fill="both", expand=True)

        canvas = tk.Canvas(wrapper, bg="#edf4f7", highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(wrapper, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        card = ttk.Frame(canvas, style="AboutCard.TFrame", padding=24)
        card.columnconfigure(0, weight=1)
        canvas_window = canvas.create_window((0, 0), window=card, anchor="nw")

        def update_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(canvas_window, width=canvas.winfo_width())

        card.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", update_scroll_region)

        
        ttk.Label(card, text="", style="AboutAccent.TLabel", anchor="center", justify="center").grid(row=1, column=0, pady=(0, 18))

        bio_frame = ttk.Frame(card, style="AboutInner.TFrame", padding=18)
        bio_frame.grid(row=2, column=0, sticky="ew", pady=(0, 18))
        bio_frame.columnconfigure(0, weight=1)
        ttk.Label(
            bio_frame,
            text=(
               "This project aims to build an instant translator for American Sign Language"

" Using a camera, computer vision, and machine learning technologies, it will recognize"
" hand gestures and convert them into text in real time, with the potential for future expansion to add"
" more gestures and improve the user experience."
            ),
            style="AboutBio.TLabel",
            anchor="center",
            justify="center",
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(card, text="إعداد الفريق", style="AboutTitle.TLabel", anchor="center", justify="center").grid(row=0, column=0, pady=(0, 12))
        team_names = [
            "أحمد يوسف يوسف منصور بوشه",
            "ابانوب امير جبران جرجس جبران",
            "محمود السيد على السعيد الترابى",
            "انس محمد سعيد محمد عفيفى",
        ]

        names_frame = ttk.Frame(card, style="AboutInner.TFrame", padding=16)
        names_frame.grid(row=3, column=0, sticky="ew", pady=(0, 18))
        names_frame.columnconfigure(0, weight=1)

        for index, name in enumerate(team_names):
            ttk.Label(
                names_frame,
                text=name,
                style="AboutName.TLabel",
                anchor="center",
                justify="center",
            ).grid(row=index * 2, column=0, pady=6, sticky="ew")
            if index < len(team_names) - 1:
                ttk.Separator(names_frame, orient="horizontal").grid(
                    row=index * 2 + 1,
                    column=0,
                    sticky="ew",
                    pady=3,
                )

        ttk.Label(
            card,
            text="قسم حاسبات وتحكم آلي",
            style="AboutAccent.TLabel",
            anchor="center",
            justify="center",
        ).grid(row=4, column=0, pady=(4, 8))
        ttk.Label(
            card,
            text="مستوى اول",
            style="AboutMeta.TLabel",
            anchor="center",
            justify="center",
        ).grid(row=5, column=0, pady=3)
        ttk.Label(
            card,
            text="كلية الهندسة - جامعة طنطا",
            style="AboutMeta.TLabel",
            anchor="center",
            justify="center",
        ).grid(row=6, column=0, pady=(3, 6))

    def _build_about_tab_v4(self) -> None:
        self.style.configure("AboutCard.TFrame", background="#f7f1ea")
        self.style.configure("AboutInner.TFrame", background="#fffdf9")
        self.style.configure("AboutTitle.TLabel", background="#f7f1ea", foreground="#123047", font=("Segoe UI", 22, "bold"))
        self.style.configure("AboutSection.TLabel", background="#f7f1ea", foreground="#0f766e", font=("Tahoma", 17, "bold"))
        self.style.configure("AboutName.TLabel", background="#fffdf9", foreground="#18415d", font=("Tahoma", 14, "bold"))
        self.style.configure("AboutMeta.TLabel", background="#f7f1ea", foreground="#49657a", font=("Tahoma", 13, "bold"))
        self.style.configure("AboutAccent.TLabel", background="#f7f1ea", foreground="#0f766e", font=("Tahoma", 16, "bold"))
        self.style.configure("AboutBio.TLabel", background="#fffdf9", foreground="#3c5568", font=("Segoe UI", 12), wraplength=900, justify="center")

        for child in self.about_tab.winfo_children():
            child.destroy()

        wrapper = ttk.Frame(self.about_tab, padding=18)
        wrapper.pack(fill="both", expand=True)

        canvas = tk.Canvas(wrapper, bg="#edf4f7", highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(wrapper, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        card = ttk.Frame(canvas, style="AboutCard.TFrame", padding=24)
        card.columnconfigure(0, weight=1)
        canvas_window = canvas.create_window((0, 0), window=card, anchor="nw")

        def update_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(canvas_window, width=canvas.winfo_width())

        card.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", update_scroll_region)

        ttk.Label(card, text="About The Project", style="AboutTitle.TLabel", anchor="center", justify="center").grid(row=0, column=0, pady=(0, 10))
        ttk.Label(card, text="مشروع مترجم لغة الإشارة", style="AboutAccent.TLabel", anchor="center", justify="center").grid(row=1, column=0, pady=(0, 18))

        bio_frame = ttk.Frame(card, style="AboutInner.TFrame", padding=18)
        bio_frame.grid(row=2, column=0, sticky="ew", pady=(0, 18))
        bio_frame.columnconfigure(0, weight=1)
        ttk.Label(
            bio_frame,
            text=(
                "This project aims to build an instant translator for American Sign Language using "
                "a camera, computer vision, and machine learning technologies. It will recognize "
                "hand gestures and convert them into text in real time, with the potential for "
                "future expansion to add more gestures and improve the user experience."
            ),
            style="AboutBio.TLabel",
            anchor="center",
            justify="center",
        ).grid(row=0, column=0, sticky="ew")

        ttk.Label(card, text="إعداد الفريق", style="AboutSection.TLabel", anchor="center", justify="center").grid(row=3, column=0, pady=(0, 14))

        team_names = [
            "أحمد يوسف يوسف منصور بوشه",
            "ابانوب امير جبران جرجس جبران",
            "محمود السيد على السعيد الترابى",
            "انس محمد سعيد محمد عفيفى", 
        ]

        names_frame = ttk.Frame(card, style="AboutInner.TFrame", padding=16)
        names_frame.grid(row=4, column=0, sticky="ew", pady=(0, 18))
        names_frame.columnconfigure(0, weight=1)

        for index, name in enumerate(team_names):
            ttk.Label(
                names_frame,
                text=name,
                style="AboutName.TLabel",
                anchor="center",
                justify="center",
            ).grid(row=index * 2, column=0, pady=6, sticky="ew")
            if index < len(team_names) - 1:
                ttk.Separator(names_frame, orient="horizontal").grid(
                    row=index * 2 + 1,
                    column=0,
                    sticky="ew",
                    pady=3,
                )

        ttk.Label(card, text="قسم حاسبات وتحكم آلي", style="AboutAccent.TLabel", anchor="center", justify="center").grid(row=5, column=0, pady=(4, 8))
        ttk.Label(card, text="مستوى اول", style="AboutMeta.TLabel", anchor="center", justify="center").grid(row=6, column=0, pady=3)
        ttk.Label(card, text="كلية الهندسة - جامعة طنطا", style="AboutMeta.TLabel", anchor="center", justify="center").grid(row=7, column=0, pady=(3, 6))

    def _build_translate_tab(self) -> None:
        wrapper = ttk.Frame(self.translate_tab, padding=16)
        wrapper.pack(fill="both", expand=True)
        wrapper.columnconfigure(0, weight=1)
        wrapper.rowconfigure(0, weight=1)

        content = ttk.Frame(wrapper)
        content.grid(row=0, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        main_panel = ttk.Frame(content)
        main_panel.grid(row=0, column=0, sticky="nsew")
        main_panel.columnconfigure(0, weight=0)
        main_panel.columnconfigure(1, weight=1)
        main_panel.rowconfigure(0, weight=1)

        controls = ttk.Frame(main_panel, style="Card.TFrame", padding=18)
        controls.grid(row=0, column=0, sticky="nsw")

        ttk.Label(controls, text="Live Translator", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(controls, text="Auto-write only happens after stability and 80% confidence.", style="CardBody.TLabel", wraplength=280, justify="left").pack(anchor="w", pady=(8, 16))

        self.translator_mode_var = tk.StringVar(value="Static signs")
        ttk.Label(controls, text="Mode", style="CardBody.TLabel").pack(anchor="w", pady=(4, 6))
        ttk.Combobox(
            controls,
            textvariable=self.translator_mode_var,
            values=["Static signs", "Dynamic signs", "Hybrid mode"],
            state="readonly",
            width=24,
        ).pack(fill="x", pady=(0, 12))

        self.start_cam_button = tk.Button(controls, text="Start Camera", command=self.start_translator, bg="#0f766e", fg="white", font=("Segoe UI", 11, "bold"), relief="flat", padx=16, pady=9)
        self.stop_cam_button = tk.Button(controls, text="Stop Camera", command=self.stop_translator, bg="#b91c1c", fg="white", font=("Segoe UI", 11, "bold"), relief="flat", padx=16, pady=9)
        self.clear_button = tk.Button(controls, text="Clear Sentence", command=self.clear_sentence, bg="#475569", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=16, pady=9)
        self.remove_button = tk.Button(controls, text="Remove Last", command=self.remove_last_word, bg="#7c3aed", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=16, pady=9)
        self.accept_button = tk.Button(controls, text="Accept Current", command=self.accept_current, bg="#d97706", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=16, pady=9)

        self.start_cam_button.pack(fill="x", pady=4)
        self.stop_cam_button.pack(fill="x", pady=4)
        self.clear_button.pack(fill="x", pady=4)
        self.remove_button.pack(fill="x", pady=4)
        self.accept_button.pack(fill="x", pady=4)
        self.shortcuts_var = tk.StringVar(value="Shortcuts: Q stop camera | C clear sentence | Space accept current | Backspace remove last")

        video_card = ttk.Frame(main_panel, style="Card.TFrame", padding=14)
        video_card.grid(row=0, column=1, sticky="nsew", padx=(16, 0))
        video_card.columnconfigure(0, weight=1)
        video_card.rowconfigure(0, weight=1)
        video_card.rowconfigure(1, weight=0)

        self.video_label = tk.Label(
            video_card,
            bg="#cbd5e1",
            fg="#32506b",
            text=self.video_placeholder_text,
            font=("Segoe UI", 18, "bold"),
            relief="flat",
            anchor="center",
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        shortcuts_label = ttk.Label(
            video_card,
            textvariable=self.shortcuts_var,
            style="Body.TLabel",
            justify="left",
            wraplength=860,
        )
        shortcuts_label.grid(row=1, column=0, sticky="w", pady=(12, 0))

    def start_collection(self) -> None:
        if self.collect_thread and self.collect_thread.is_alive():
            return

        collection_mode = self.collection_mode_var.get().strip()
        custom_label = self.custom_class_var.get().strip()
        if collection_mode == "Dynamic signs":
            label = custom_label if custom_label else self.dynamic_class_var.get().strip()
            default_save_dir = SEQUENCE_DATASET_DIR
        else:
            label = custom_label if custom_label else self.class_var.get().strip()
            default_save_dir = RAW_DATASET_DIR
        samples_text = self.samples_var.get().strip()
        save_dir_value = self.save_dir_var.get().strip()
        save_dir = Path(save_dir_value) if save_dir_value else default_save_dir
        if not label:
            messagebox.showerror("Invalid Input", "Please choose or type a class label.")
            return
        if not samples_text.isdigit():
            messagebox.showerror("Invalid Input", "Samples must be a positive integer.")
            return

        samples = int(samples_text)
        self.collect_button.config(state="disabled")
        self.collect_status_var.set(f"Collecting {samples} items for {label}...")

        def worker() -> None:
            try:
                if collection_mode == "Dynamic signs":
                    collect_sequence_data(label, samples, save_dir)
                else:
                    collect_samples_for_class(label, samples, save_dir)
                self.root.after(0, lambda: self.collect_status_var.set(f"Finished collecting data for {label}."))
            except Exception as error:
                self.root.after(0, lambda: messagebox.showerror("Collection Error", str(error)))
                self.root.after(0, lambda: self.collect_status_var.set("Collection failed."))
            finally:
                self.root.after(0, lambda: self.collect_button.config(state="normal"))

        self.collect_thread = threading.Thread(target=worker, daemon=True)
        self.collect_thread.start()

    def start_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            return

        self.train_button.config(state="disabled")
        self.log_text.delete("1.0", tk.END)
        training_mode = self.training_mode_var.get().strip()

        def worker() -> None:
            stream = io.StringIO()
            try:
                with redirect_stdout(stream):
                    if training_mode == "Dynamic model":
                        train_sequence_model()
                    else:
                        train_model()
            except Exception as error:
                stream.write(f"\nError: {error}\n")
            finally:
                self.log_queue.put(stream.getvalue())
                self.log_queue.put("__TRAINING_DONE__")

        self.training_thread = threading.Thread(target=worker, daemon=True)
        self.training_thread.start()

    def _poll_logs(self) -> None:
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message == "__TRAINING_DONE__":
                    self.train_button.config(state="normal")
                else:
                    self.log_text.insert(tk.END, message)
                    self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_logs)

    def start_translator(self) -> None:
        if self.translator_running:
            return
        try:
            mode_name = self.translator_mode_var.get().strip().lower()
            if mode_name == "dynamic signs":
                self.translator_session = DynamicTranslatorSession()
            elif mode_name == "hybrid mode":
                self.translator_session = HybridTranslatorSession()
            else:
                self.translator_session = TranslatorSession()
            self.translator_session.start()
        except Exception as error:
            messagebox.showerror("Translator Error", str(error))
            return
        self.translator_running = True
        self.root.focus_force()
        self.video_label.focus_force()
        self._update_translator_frame()

    def stop_translator(self) -> None:
        self.translator_running = False
        if self.translator_session is not None:
            self.translator_session.stop()
        self.video_label.configure(image="", text=self.video_placeholder_text)
        self.video_label.image = None

    def clear_sentence(self) -> None:
        if self.translator_session is not None:
            self.translator_session.clear_sentence()

    def remove_last_word(self) -> None:
        if self.translator_session is not None:
            self.translator_session.remove_last_word()

    def accept_current(self) -> None:
        if self.translator_session is not None:
            self.translator_session.accept_current_prediction()

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Key>", self._handle_global_keypress, add="+")

    def _handle_global_keypress(self, event) -> str | None:
        if not self.translator_running:
            return None

        keysym = (event.keysym or "").lower()
        if keysym == "q":
            self.stop_translator()
            return "break"
        if keysym == "c":
            self.clear_sentence()
            return "break"
        if keysym == "space":
            self.accept_current()
            return "break"
        if keysym == "backspace":
            self.remove_last_word()
            return "break"
        return None

    def _update_translator_frame(self) -> None:
        if not self.translator_running:
            return

        try:
            result = self.translator_session.read_result()
            if result.success and result.frame is not None:
                draw_translator_overlay(
                    result.frame,
                    result.current_prediction,
                    result.confidence,
                    result.sentence_words,
                    result.stable_label,
                    result.info_message,
                )
                rgb_frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                image = self._fit_image_to_label(image)
                tk_image = ImageTk.PhotoImage(image=image)
                self.video_label.configure(image=tk_image, text="")
                self.video_label.image = tk_image
            else:
                self.video_label.configure(image="", text="Camera frame is not available")
                self.video_label.image = None
        except Exception as error:
            self.translator_running = False
            self.translator_session.stop()
            self.video_label.configure(image="", text=f"Camera error: {error}")
            self.video_label.image = None
            return

        self.root.after(15, self._update_translator_frame)

    def _fit_image_to_label(self, image: Image.Image) -> Image.Image:
        self.root.update_idletasks()
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()

        if label_width < 100 or label_height < 100:
            label_width = 920
            label_height = 620

        image_width, image_height = image.size
        scale = min(label_width / image_width, label_height / image_height)
        scale = max(scale, 0.1)

        resized_width = max(1, int(image_width * scale))
        resized_height = max(1, int(image_height * scale))
        return image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    def on_close(self) -> None:
        self.stop_translator()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SignLanguageGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
