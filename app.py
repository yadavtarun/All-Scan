from flask import Flask, render_template, redirect,send_from_directory
from flask import request
import warnings
import cv2
import os
from PIL import Image
from skimage.filters import threshold_local
import pytesseract as tess
import nltk
import re
from bs4 import BeautifulSoup as bs
import urllib
from pyzbar import pyzbar
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__, template_folder='C:\\Users\\HP\\PycharmProjects\\Allscan\\templates')
global cur_data
cur_path = "C:\\Users\\HP\\PycharmProjects\\Allscan\\imagetopdf\\images1"


def image_to_pdf(mode="pdf"):
    # print(cur_path)
    files_in_dir = os.listdir(cur_path)
    conventions = ['png', 'jpg', 'jpeg', 'jfif']
    img_names = []
    # print(files_in_dir)
    for file in files_in_dir:
        ext = file.split('.')[-1]
        if ext in conventions:
            img_names.insert(0, file)
    # print(img_names)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    img_read = []
    for name in img_names:
        name1 = cur_path + "\\" + name
        # print(name1)
        img = cv2.imread(name1)
        img_read.insert(0, img)
    # print(img)
    thsh_img = []
    for img in img_read:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
        img_gray = clahe.apply(img_gray)
        ret, th = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)
        thsh_img.append(th)
    # cv2.imshow('image',th)
    # cv2.waitKey(50)

    # Digitise image in black and white as a scanned document
    digitised_image_names = []
    for ind in range(len(img_read)):
        img_gray = cv2.cvtColor(img_read[ind].copy(), cv2.COLOR_BGR2GRAY)
        th = threshold_local(img_gray.copy(), 101, offset=10, method="gaussian")
        img_gray = (img_gray > th)
        imgg = Image.fromarray(img_gray)
        size = (img_read[ind].shape[0], img_read[ind].shape[1])
        imgg.resize(size)
        name = cur_path + "\\digitised_" + img_names[ind].split('.')[0] + '.jpg'
        digitised_image_names.append(name)
        imgg.save(digitised_image_names[ind])

    if mode == "pdf":
        # Convert all digitised images to pdf format
        digitised_images = []

        for name in digitised_image_names:
            imgg = Image.open(name)
            digitised_images.append(imgg)
            name = cur_path + "\\digitised_images" + '.pdf'
            if len(digitised_images) > 1:
                digitised_images[0].save(name, save_all=True, append_images=digitised_images[1:], resolution=100.0)
            else:
                digitised_images[0].save(name)

        for file in digitised_image_names:
            os.remove(file)
        files = os.listdir(cur_path)

        for file1 in files:
            if file1.split('.')[-1] in conventions:
                os.remove(cur_path + "\\" + file1)
        curr_path = os.getcwd()
        files1 = os.listdir(curr_path)
        for file in files1:
            if file.split('.')[-1] in conventions:
                os.remove(curr_path + "\\" + file)
        return

    elif mode == "text":
        # create text file
        name = cur_path + '\\text.txt'
        txt_file = open(name, "w", encoding="utf-8")
        # Extract text from image using PyOcr
        tess.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
        for i, name in enumerate(digitised_image_names):
            txt = tess.image_to_string(img)
            txt = ' '.join(txt.replace('-\n', '').replace('\n', ' ').split())
            txt_file.write("[Image " + str(i + 1) + " text]\n\n")
            txt_file.write(txt)
            txt_file.write("\n\n")

        txt_file.close()
        for file in digitised_image_names:
            os.remove(file)
        files = os.listdir(cur_path)

        for file1 in files:
            if file1.split('.')[-1] in conventions:
                os.remove(cur_path + "\\" + file1)
        curr_path = os.getcwd()
        files1 = os.listdir(curr_path)
        for file in files1:
            if file.split('.')[-1] in conventions:
                os.remove(curr_path + "\\" + file)
        return


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['jpeg', 'jpg', 'png', 'jfif']


@app.route('/uploadimages', methods=['POST', 'GET'])
def uploadimages():
    file_names = []
    curr_path = os.getcwd()
    files_in_dir = os.listdir(cur_path)
    for file in file_names:
        if file.split('.')[-1] in ['jpeg', 'png', 'jpg', 'pdf', 'jfif']:
            os.remove(file)
    uploaded_files = request.files.getlist("files")

    UPLOAD_FOLDER = 'images1'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    for file in uploaded_files:
        if file.filename.split('.')[-1] in ['jpeg', 'png', 'jpg', 'jfif']:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    image_to_pdf(mode="pdf")

    return send_from_directory(cur_path, 'digitised_images.pdf', as_attachment=True)


@app.route('/uploadimage', methods=['POST', 'GET'])
def uploadimage():
    file_names = []
    files_in_dir = os.listdir(cur_path)
    for file in file_names:
        if file.split('.')[-1] in ['jpeg', 'png', 'jpg', 'pdf', 'txt', 'jfif']:
            os.remove(file)

    uploaded_files = request.files.getlist("files")

    UPLOAD_FOLDER = 'images1'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    for file in uploaded_files:
        if file.filename.split('.')[-1] in ['jpeg', 'png', 'jpg', 'jfif']:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    image_to_pdf(mode="text")

    return send_from_directory(cur_path, 'text.txt', as_attachment=True)


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    text = request.form['text']
    file_names = []
    files_in_dir = os.listdir(cur_path)

    text = re.sub(r'\[[0-9]*\]', ' ', text)  # removes numbers and boxes
    text = re.sub(r'\s+', ' ', text)  # removes spaces

    print(text)
    sen = text.split(".")
    sentence_tokens = []
    print(sen)
    for i in sen:
        sentence_tokens.append(i + ".")
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # print(cleaned_data)
    # print(len(sentence_tokens))

    # Tokenization of words
    words_tokens = nltk.word_tokenize(text)

    # Calculate the frequency
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    # print(words_tokens)
    for word in words_tokens:
        if word not in stopwords:
            if word not in word_frequencies:
                word_frequencies[word] = words_tokens.count(word)

    #        print(word_frequencies)

    # Calculate weighted frequency
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    sentence_score = {}
    for sentence in sentence_tokens:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies:
                # if (len(sentence.split(" ")))<50:
                if sentence not in sentence_score:
                    sentence_score[sentence] = word_frequencies[word]
                else:
                    sentence_score[sentence] += word_frequencies[word]
    print(sentence_score)

    import heapq
    len(sentence_score)

    fin_summary = ""
    n = int(0.4 * len(sentence_score))
    summary = heapq.nlargest(n, sentence_score, key=sentence_score.get)
    for i in summary:
        fin_summary += i
    print(fin_summary)
    name1 = cur_path + '\\summarize.txt'
    txt_file1 = open(name1, "w", encoding="utf-8")
    txt_file1.write(fin_summary)
    txt_file1.write("\n\n")

    txt_file1.close()

    # print(fin_summary)

    return send_from_directory(cur_path, 'summarize.txt', as_attachment=True)


@app.route('/submit1', methods=['POST', 'GET'])
def submit1():
    text1 = request.form['text']
    file_names = []
    files_in_dir = os.listdir(cur_path)

    url = ""
    url += text1

    htmlDoc = urllib.request.urlopen(url)

    soupObject = bs(htmlDoc, "html.parser")
    paragraphContents = soupObject.findAll("p")
    # print(paragraphContents)

    # Data cleaning
    allparagraphContents = ""

    for para in paragraphContents:
        allparagraphContents += para.text  # convirting html tags to text
    # print(allparagraphContents)
    text = re.sub(r'\[[0-9]*\]', ' ', allparagraphContents)  # removes numbers and boxes
    text = re.sub(r'\s+', ' ', text)  # removes spaces

    print(text)
    sen = text.split(".")
    sentence_tokens = []
    print(sen)
    for i in sen:
        sentence_tokens.append(i + ".")
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # print(cleaned_data)
    # print(len(sentence_tokens))

    # Tokenization of words
    words_tokens = nltk.word_tokenize(text)

    # Calculate the frequency
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    # print(words_tokens)
    for word in words_tokens:
        if word not in stopwords:
            if word not in word_frequencies:
                word_frequencies[word] = words_tokens.count(word)

    #        print(word_frequencies)

    # Calculate weighted frequency
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    sentence_score = {}
    for sentence in sentence_tokens:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies:
                # if (len(sentence.split(" ")))<50:
                if sentence not in sentence_score:
                    sentence_score[sentence] = word_frequencies[word]
                else:
                    sentence_score[sentence] += word_frequencies[word]
    print(sentence_score)

    import heapq
    len(sentence_score)

    fin_summary = ""
    n = int(0.2 * len(sentence_score))
    summary = heapq.nlargest(n, sentence_score, key=sentence_score.get)
    for i in summary:
        fin_summary += i
    print(fin_summary)
    name1 = cur_path + '\\summarize.txt'
    txt_file1 = open(name1, "w", encoding="utf-8")
    txt_file1.write(fin_summary)
    txt_file1.write("\n\n")

    txt_file1.close()

    # print(fin_summary)

    return send_from_directory(cur_path, 'summarize.txt', as_attachment=True)


@app.route('/textsummarize1', methods=['POST', 'GET'])
def textsummarize1():
    file_names = []
    files_in_dir = os.listdir(cur_path)

    name = cur_path + '\\text.txt'
    txt_file = open(name, "r", encoding="utf-8")

    text = txt_file.read()
    print(text)

    text = re.sub(r'\[[0-9]*\]', ' ', text)  # removes numbers and boxes
    text = re.sub(r'\s+', ' ', text)  # removes spaces

    print(text)
    txt_file.close()
    sen = text.split(".")
    sentence_tokens = []
    print(sen)
    for i in sen:
        sentence_tokens.append(i + ".")
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # print(cleaned_data)
    # print(len(sentence_tokens))

    # Tokenization of words
    words_tokens = nltk.word_tokenize(text)

    # Calculate the frequency
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    # print(words_tokens)
    for word in words_tokens:
        if word not in stopwords:
            if word not in word_frequencies:
                word_frequencies[word] = words_tokens.count(word)

    #        print(word_frequencies)

    # Calculate weighted frequency
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    sentence_score = {}
    for sentence in sentence_tokens:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies:
                # if (len(sentence.split(" ")))<50:
                if sentence not in sentence_score:
                    sentence_score[sentence] = word_frequencies[word]
                else:
                    sentence_score[sentence] += word_frequencies[word]
    print(sentence_score)

    import heapq
    len(sentence_score)

    fin_summary = ""
    n = int(0.20 * len(sentence_score))
    summary = heapq.nlargest(n, sentence_score, key=sentence_score.get)
    for i in summary:
        fin_summary += i
    print(fin_summary)
    name1 = cur_path + '\\summarize.txt'
    txt_file1 = open(name1, "w", encoding="utf-8")
    txt_file1.write(fin_summary)
    txt_file1.write("\n\n")

    txt_file1.close()

    # print(fin_summary)

    return send_from_directory(cur_path, 'summarize.txt', as_attachment=True)






@app.route('/qrcode', methods=['POST', 'GET'])
def qrcode():
    file_names = []
    found = set()
    curr_path = os.getcwd()
    files_in_dir = os.listdir(cur_path)
    for file in file_names:
        if file.split('.')[-1] in ['jpeg', 'png', 'jpg', 'pdf', 'jfif']:
            os.remove(file)
    uploaded_files = request.files.getlist("files")

    UPLOAD_FOLDER = 'images1'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    for file in uploaded_files:
        if file.filename.split('.')[-1] in ['jpeg', 'png', 'jpg', 'jfif']:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    files_in_dir = os.listdir(cur_path)
    conventions = ['png', 'jpg', 'jpeg', 'jfif']
    img_names = []
    # print(files_in_dir)
    for file in files_in_dir:
        ext = file.split('.')[-1]
        if ext in conventions:
            img_names.insert(0, file)
    # print(img_names)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    img_read = []
    for name in img_names:
        name1 = cur_path + "\\" + name
        # print(name1)
        img = cv2.imread(name1)
        img_read.insert(0, img)
    # print(img)
    barcodes = pyzbar.decode(img_read[0])
    for barcode in barcodes:
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        if barcodeData not in found:
            print(barcodeData)
            text = str(barcodeData)
            found.add(barcodeData)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    for file1 in img_names:
        if file1.split('.')[-1] in conventions:
            os.remove(cur_path + "\\" + file1)

    return redirect(text, code=302)



@app.route('/go')
def go():
    name1 = cur_path + '\\qr.txt'
    txt_file1 = open(name1, "r", encoding="utf-8")
    text = txt_file1.read()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    txt_file1.close()
    return redirect(text, code=302)


@app.route("/")
@app.route('/imagetotext')
def imagetotext():
    return render_template('imagetotext.html')


@app.route('/imagetopdf')
def imagetopdf():
    return render_template('imagetopdf.html')


@app.route('/tsum')
def tsum():
    return render_template('tsum.html')


@app.route('/qr')
def qr():
    return render_template("qr.html")


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)