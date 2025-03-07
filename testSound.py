import pyttsx3  # For text-to-speech

speakAlert = pyttsx3.init()
def speak_alert(message):
    """Speak the alert message aloud."""
    speakAlert.setProperty('rate', 125)
    volume = speakAlert.getProperty('volume')  
    print(f'Current volume level: {volume}')    
    voices = speakAlert.getProperty('voices')
    speakAlert.setProperty('voice',voices[0].id)
    speakAlert.say(message)
    speakAlert.runAndWait()
    
    
speak_alert("Hello from test Sound!!")