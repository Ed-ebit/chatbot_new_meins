import json

class file_handling:
    """Klasse, die das Gedächtnis des Chatbots enthaelt.
    Bekannte Eingaben sind ebenso vorhanden wie die dem Bot möglichen Ausgaben"""
    
    def __init__(self, filepath):
        try:
            with open(filepath) as antworten:
                self.antworten = json.load(antworten)
        except FileNotFoundError:
            print("Konnte keine Bibliothek laden.")  
        
        self.reaktionen = self.antworten['reaktionen']     
        self.zufallsantworten = self.antworten['zufallsantworten']
        
    
    # def getAntworten(self, type):
    #     if type == "zufall":
    #         return self.zufallsantworten
    #     elif type == "reaktion":
    #         return self.reaktionen
    #     else:
    #         return "Fehler"


                
