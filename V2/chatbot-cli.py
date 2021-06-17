#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from chatbot import Chatbot
from file_handling import file_handling


def main():
    # Listen
    file = file_handling("answers.json")
    zufallsantworten = file.zufallsantworten
    reaktionen = file.reaktionen

    # Ausgabe Begrüßung
    print("Willkommen beim Chatbot (v2)")
    print("Zum Beenden geben Sie bye ein...")
    print("Worüber wollen Sie sprechen?")
    print("")

    # Chatbot-Objekt
    bot = Chatbot(reaktionen, zufallsantworten)

    # Logik
    nutzereingabe = ""
    while nutzereingabe != "bye":
        nutzereingabe = ""
        while nutzereingabe == "":
            nutzereingabe = input("Ihre Frage oder Antwort: ")
        if nutzereingabe == "bye":
            break
        bot.set_Message(nutzereingabe)
        print(bot.get_Response())

    # Ausgabe Verabschiedung
    print("Bis zum nächsten Mal.")


main()
