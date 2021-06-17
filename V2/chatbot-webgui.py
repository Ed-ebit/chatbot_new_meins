#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, flash
from chatbot import Chatbot
from file_handling import file_handling

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Pa$$w0rd'


@app.route('/', methods=('GET', 'POST'))
def index():
    file = file_handling("answers.json")
    zufallsantworten = file.zufallsantworten
    reaktionen = file.reaktionen
    chatbot_label = "Hallo, worüber wollen Sie sprechen?"

    # Listen
    zufallsantworten = file.zufallsantworten
    reaktionen = file.reaktionen
    # Logik
    if request.method == 'POST':
        chatbot_input = request.form['chatbot_input']
        if not chatbot_input:
            flash("Ohne Frage kann ich nicht antworten")
        else:
            bot = Chatbot(reaktionen, zufallsantworten)
            bot.set_Message(chatbot_input)
            chatbot_label = bot.get_Response()

    return render_template('chatbot.html', chatbot_label=chatbot_label)
