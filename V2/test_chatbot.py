import unittest
from chatbot import Chatbot


class TestChatBot(unittest.TestCase):
    def test_intelligent_answers(self):
        """Test der intelligenten Antworten"""
        self.__zufallsantworten = ["Oh wirklich...", "Interessant", "Das kann man so sehen.", "Ich verstehe..."]
        self.__reaktionen = {"hallo": "aber hallo",
                             "geht": "Was verstehst Du darunter",
                             "schmeckt": "Ich habe keinen Geschmackssinn"}
        __data = ["hallo du", "geht es dir gut", "schmeckt die Suppe"]
        __bot = Chatbot(self.__reaktionen, self.__zufallsantworten)
        for sentence in __data:
            __bot.set_Message(sentence)
            self.__response = __bot.get_Response()
            __words = sentence.split()
            for word in __words:
                if word in self.__reaktionen:
                    self.__rightResponse = self.__reaktionen[word]
            self.assertEqual(self.__response, self.__rightResponse)

    def test_random_answers(self):
        """Test der Zufallsantworten"""
        self.__zufallsantworten = ["Oh wirklich...", "Interessant", "Das kann man so sehen.", "Ich verstehe..."]
        self.__reaktionen = {"hallo": "aber hallo",
                             "geht": "Was verstehst Du darunter",
                             "schmeckt": "Ich habe keinen Geschmackssinn"}
        __data = ["niemals", "Ich weiß garnicht ob Sie's schon wussten", "der ist ja doof"]
        __bot = Chatbot(self.__reaktionen, self.__zufallsantworten)
        for sentence in __data:
            __bot.set_Message(sentence)
            self.__response = __bot.get_Response()
            self.assertIn(self.__response, self.__zufallsantworten)

    def test_special_characters(self):
        """Test von Sonderzeichen"""
        self.__zufallsantworten = ["Oh wirklich...", "Interessant", "Das kann man so sehen.", "Ich verstehe..."]
        self.__reaktionen = {"hallo": "aber hallo",
                             "geht": "Was verstehst Du darunter",
                             "schmeckt": "Ich habe keinen Geschmackssinn"}
        __data = ["/", "#/$", "=()&"]
        __bot = Chatbot(self.__reaktionen, self.__zufallsantworten)
        for sentence in __data:
            __bot.set_Message(sentence)
            self.__response = __bot.get_Response()
            self.assertIn(self.__response, self.__zufallsantworten)
            
    def test_empty_spaces(self):
        """Test von Leerzeichen"""
        self.__zufallsantworten = ["Oh wirklich...", "Interessant", "Das kann man so sehen.", "Ich verstehe..."]
        self.__reaktionen = {"hallo": "aber hallo",
                             "geht": "Was verstehst Du darunter",
                             "schmeckt": "Ich habe keinen Geschmackssinn"}
        __data = [" "]        
        __bot = Chatbot(self.__reaktionen, self.__zufallsantworten)
        for sentence in __data:
            __bot.set_Message(sentence)
            self.__response = __bot.get_Response()
            self.assertIn(self.__response, self.__zufallsantworten)


if __name__ == '__main__':
    unittest.main(verbosity=2)
