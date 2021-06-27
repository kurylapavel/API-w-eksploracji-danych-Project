from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.core.window import Window

import os
import re
import pandas as pd

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#region Window config
Window.size = (1176, 450)
#endregion

#region Global variables
path = ""
labelPath = Label()
#endregion

#region Classes
class MyFileChooser(FileChooserListView):
    def on_submit(self, *args):
        fp=args[0][0]
        global path
        global labelPath
        labelPath.text = args[0][0]
        path = args[0][0]
        self.parent.parent.parent.dismiss()

class MainScreen(FloatLayout):

    #region Methods
    def filebtn(self, instance):
        self.popup = Popup(title='Select File',
                      content=MyFileChooser(),
                      size_hint=(None, None), size=(400, 400))
        self.popup.open()

    def checkErrors(self):
        global path
        if path == "":
            error = Popup(title='Error',
                          content=Label(text='Choose file before start!\n(Click outside the error to close)'),
                          auto_dismiss=True, size_hint=(.4, .3))
            error.open()
            return True
        elif path[-4:] != ".txt":
            error = Popup(title='Error',
                          content=Label(text='Only txt is supported!\n(Click outside the error to close)'),
                          auto_dismiss=True, size_hint=(.4, .3))
            error.open()
            return True
        elif os.stat(path).st_size == 0:
            error = Popup(title='Error', content=Label(text='File is empty\n(Click outside the error to close)'),
                          auto_dismiss=True, size_hint=(.4, .3))
            error.open()
            return True

        return False

    def readFile(self):
        # Open a file: file
        global path
        file = open(path, mode='r')
        text = file.read()
        file.close()
        return text

    def tokenizationByWords(self,instance):
        if self.checkErrors():
            return
        text = self.readFile()
        words = nltk.word_tokenize(text)
        filePath = f"{path[:-4]}_words.txt"
        with open(filePath, "w") as txt_file:
            for word in words:
                if re.match(r'\w', word):
                    txt_file.write("".join(word) + ", ")
        os.startfile(filePath)

    def tokenazitionBySentences(self,instance):
        if self.checkErrors():
            return
        text = self.readFile()
        sentences = nltk.sent_tokenize(text)
        filePath = f"{path[:-4]}_sentences.txt"
        with open(filePath, "w") as txt_file:
            for sentence in sentences:
                txt_file.write("".join(sentence) + "\n")
        os.startfile(filePath)

    def lematization(self,instance):
        if self.checkErrors():
            return
        text = self.readFile()
        words = nltk.word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        filePath = f"{path[:-4]}_lematization.txt"
        with open(filePath, "w") as txt_file:
            for word in words:
                if re.match(r'\w',word):
                    txt_file.write("".join(lemmatizer.lemmatize(word=word, pos=wordnet.VERB)) + ", ")
        os.startfile(filePath)

    def stemmer(self,instance):
        if self.checkErrors():
            return
        text = self.readFile()
        words = nltk.word_tokenize(text)
        stemmer = PorterStemmer()
        filePath = f"{path[:-4]}_stemmer.txt"
        with open(filePath, "w") as txt_file:
            for word in words:
                if re.match(r'\w',word):
                    txt_file.write("".join(stemmer.stem(word)) + ", ")
        os.startfile(filePath)

    def stopWords(self,instance):
        nltk.download('stopwords')
        stop_words = set(stopwords.words("english"))

        if self.checkErrors():
            return
        text = self.readFile()
        words = nltk.word_tokenize(text)
        filePath = f"{path[:-4]}_stopWords.txt"
        without_stop_words = [word for word in words if not word in stop_words]

        with open(filePath, "w") as txt_file:
            for word in without_stop_words:
                txt_file.write("".join(word) + " ")
        os.startfile(filePath)

    def findMails(self,instance):
        if self.checkErrors():
            return
        text = self.readFile()
        match = re.findall(r'[\w\.-]+@[\w\.-]+', text)
        filePath = f"{path[:-4]}_mails.txt"

        with open(filePath, "w") as txt_file:
            for mail in match:
                txt_file.write("".join(mail) + "\n")
        os.startfile(filePath)

    def findPhones(self,instance):
        if self.checkErrors():
            return
        text = self.readFile()
        match = re.findall(r"[\+\d]?(\d{2,3}[-\.\s]??\d{2,3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})", text)
        filePath = f"{path[:-4]}_phones.txt"

        with open(filePath, "w") as txt_file:
            for phone in match:
                txt_file.write("".join(phone) + "\n")
        os.startfile(filePath)

    def bagOfWords(self,instance):
        if self.checkErrors():
            return
        text = self.readFile()
        sentences = nltk.sent_tokenize(text)
        filePath = f"{path[:-4]}_bagOfWords.csv"

        count_vectorizer = CountVectorizer()
        bag_of_words = count_vectorizer.fit_transform(sentences)
        feature_names = count_vectorizer.get_feature_names()
        df = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)
        df.to_csv(filePath,index=False)
        os.startfile(filePath)

    def tfIdf(self,instance):
        if self.checkErrors():
            return
        text = self.readFile()
        sentences = nltk.sent_tokenize(text)
        tfidf_vectorizer = TfidfVectorizer()
        values = tfidf_vectorizer.fit_transform(sentences)
        filePath = f"{path[:-4]}_tfidf.csv"
        feature_names = tfidf_vectorizer.get_feature_names()
        df = pd.DataFrame(values.toarray(), columns=feature_names)

        df.to_csv(filePath,index=False)
        os.startfile(filePath)
    #endregion


    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.orientation = 'vertical'

        #region Initialization elements
        self.btnfile = Button(text='Open File',size_hint=(0.3,0.15),pos=(408,350))

        global labelPath
        labelPath = Label(text='Choose file..',size_hint=(0.3,0.15),pos=(792,350))

        self.btnWordTokenization = Button(text='Tokenazition by words',size_hint=(0.3,0.15),pos=(24,250))
        self.btnSentencesTokenization = Button(text='Tokenazition by sentences',size_hint=(0.3,0.15),pos=(408,250))
        self.btnLematization = Button(text='Lemmatization',size_hint=(0.3,0.15),pos=(792,250))

        self.btnStemmer = Button(text='Stemmer',size_hint=(0.3,0.15),pos=(24,150))
        self.btnStopWords = Button(text='Stop words',size_hint=(0.3,0.15),pos=(408,150))
        self.btnFindMails = Button(text='Find all mails',size_hint=(0.3,0.15),pos=(792,150))

        self.btnFindPhones = Button(text='Find all phones',size_hint=(0.3,0.15),pos=(24,50))
        self.btnBagOfWords = Button(text='Bag of words',size_hint=(0.3,0.15),pos=(408,50))
        self.btnTfIdf = Button(text='TF-IDF',size_hint=(0.3,0.15),pos=(792,50))
        #endregion

        #region Binding events
        self.btnfile.bind(on_press=self.filebtn)
        self.btnWordTokenization.bind(on_press=self.tokenizationByWords)
        self.btnSentencesTokenization.bind(on_press=self.tokenazitionBySentences)
        self.btnLematization.bind(on_press=self.lematization)
        self.btnStemmer.bind(on_press=self.stemmer)
        self.btnStopWords.bind(on_press=self.stopWords)
        self.btnFindMails.bind(on_press=self.findMails)
        self.btnFindPhones.bind(on_press=self.findPhones)
        self.btnBagOfWords.bind(on_press=self.bagOfWords)
        self.btnTfIdf.bind(on_press=self.tfIdf)
        #endregion

        #region Adding widgets
        self.add_widget(self.btnfile)
        self.add_widget(labelPath)
        self.add_widget(self.btnWordTokenization)
        self.add_widget(self.btnSentencesTokenization)
        self.add_widget(self.btnLematization)
        self.add_widget(self.btnStemmer)
        self.add_widget(self.btnStopWords)
        self.add_widget(self.btnFindMails)
        self.add_widget(self.btnFindPhones)
        self.add_widget(self.btnBagOfWords)
        self.add_widget(self.btnTfIdf)
        #endregion

class MyApp(App):
    def build(self):
        self.title = 'Text Helper'
        return MainScreen()
#endregion

if __name__ == '__main__':
    MyApp().run()