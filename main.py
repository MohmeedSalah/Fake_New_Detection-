import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import preprocessing, model_selection, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def preprocess_data(news_data):
    # preprocess text data
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    news_data['title'] = news_data['title'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w.lower()) for w in word_tokenize(x) if w.lower() not in stop_words]))
    news_data['text'] = news_data['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(w.lower()) for w in word_tokenize(x) if w.lower() not in stop_words]))

    # encode target variable
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(news_data['label'])

    # create TF-IDF features
    tfidf_vect = TfidfVectorizer()
    X = tfidf_vect.fit_transform(news_data['title'] + " " + news_data['text'])

    return X, y, encoder, tfidf_vect


def train_models(X_train, y_train):
    # train classification models
    models = {
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=100),
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


def predict_labels(trained_models, X_test):
    predicted_labels = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        predicted_labels[name] = y_pred
    return predicted_labels


def evaluate_models(data_file):
    # load news dataset from CSV file
    news_data = pd.read_csv(data_file).dropna()

    # preprocess data
    X, y, encoder, tfidf_vect = preprocess_data(news_data)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    # train classification models
    trained_models = train_models(X_train, y_train)

    # predict labels of test set
    predicted_labels = predict_labels(trained_models, X_test)

    # evaluate models using accuracy scores and confusion matrices
    accuracies = {}
    confusion_matrices = {}
    for name, y_pred in predicted_labels.items():
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        confusion_matrix_values = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = confusion_matrix_values

    return accuracies, confusion_matrices, encoder.classes_


class ScrollableFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        tk.Frame.__init__(self, parent, **kwargs)

        # create canvas with scrollbar
        self.canvas = tk.Canvas(self)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="white")

        # bind canvas to scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', self.on_canvas_configure)

        # add scrollbar and frame to canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

    def on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


class NewsClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("News Classifier")
        self.data_file = None
        self.accuracies = None
        self.confusion_matrices = None
        self.classes = None

        # create main frame
        self.main_frame = ScrollableFrame(self.root, bg="white")
        self.main_frame.pack(side="top", fill="both", expand=True)

        # create widgets
        self.select_file_button = tk.Button(self.main_frame.scrollable_frame, text="Select Data File", command=self.select_data_file)
        self.select_file_button.pack()

        self.check_models_button = tk.Button(self.main_frame.scrollable_frame, text="Check Models", command=self.check_models, state="disabled")
        self.check_models_button.pack()

        self.accuracy_label = tk.Label(self.main_frame.scrollable_frame, text="Accuracy Scores:", font=("Arial", 12), bg="white")
        self.accuracy_label.pack(pady=10)

        self.accuracy_frame = tk.Frame(self.main_frame.scrollable_frame, bg="white")
        self.accuracy_frame.pack()

        self.confusion_matrix_label = tk.Label(self.main_frame.scrollable_frame, text="Confusion Matrices:", font=("Arial", 12), bg="white")
        self.confusion_matrix_label.pack(pady=10)

        self.confusion_matrix_frame = tk.Frame(self.main_frame.scrollable_frame, bg="white")
        self.confusion_matrix_frame.pack()

    def select_data_file(self):
        # open file dialog to select data file
        self.data_file = filedialog.askopenfilename(initialdir="./", title="Select Data File", filetypes=[("CSV Files", "*.csv")])

        # enable check button if data file is selected
        if self.data_file:
            self.check_models_button.config(state="normal")

    def check_models(self):
        # evaluate models and get results
        self.accuracies, self.confusion_matrices, self.classes = evaluate_models(self.data_file)

        # display accuracy scores
        for name, accuracy in self.accuracies.items():
            accuracy_text = f"{name}: {accuracy:.2f}"
            label = tk.Label(self.accuracy_frame, text=accuracy_text, font=("Arial", 10), bg="white")
            label.pack()

        # display confusion matrices
        for name, confusion_matrix_values in self.confusion_matrices.items():
            confusion_matrix_figure = self.plot_confusion_matrix(confusion_matrix_values, self.classes)
            canvas = FigureCanvasTkAgg(confusion_matrix_figure, master=self.confusion_matrix_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

        y_test = self.classes[self.confusion_matrices[name].sum(axis=1) > 0]
        y_pred = self.classes[self.confusion_matrices[name].sum(axis=0) > 0]
        predicted_labels = pd.DataFrame({'True Label': y_test, 'Predicted Label': y_pred})
        predicted_labels_counts = predicted_labels.groupby(['True Label', 'Predicted Label']).size().reset_index(
            name='Count')
        predicted_labels_plot = sns.catplot(x='True Label', y='Count', hue='Predicted Label',
                                            data=predicted_labels_counts, kind='bar', height=6, aspect=1.5)
        predicted_labels_plot.set_axis_labels('True Label', 'Count')
        predicted_labels_plot.ax.set_title('Predicted Labels')
        predicted_labels_plot.ax.legend(title='Predicted Label', bbox_to_anchor=(1.01, 1), loc='upper left')
        predicted_labels_canvas = FigureCanvasTkAgg(predicted_labels_plot.fig, master=self.confusion_matrix_frame)
        predicted_labels_canvas.draw()
        predicted_labels_canvas.get_tk_widget().pack()

    import seaborn as sns

    def plot_confusion_matrix(self, confusion_matrix_values, classes):
        figure = Figure(figsize=(5, 5))
        ax = figure.add_subplot(111)
        sns.heatmap(confusion_matrix_values, annot=True, cmap="Blues", fmt="d", xticklabels=classes,
                    yticklabels=classes, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        return figure


root = tk.Tk()
gui = NewsClassifierGUI(root)
root.mainloop()