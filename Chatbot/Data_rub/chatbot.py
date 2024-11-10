from tkinter import *

class Chatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot Application")
        self.root.geometry("600x400+300+100")

        # Create a Main Frame (White Color)
        main_frame = Frame(self.root, bg="white", width=600, height=400)
        main_frame.pack(fill=BOTH, expand=True)

        # Label for user input
        lbl_1_ = Label(main_frame, text="Type Something", width=20, height=2, fg='green', bg='white', font=('Arial', 14, 'bold'))
        lbl_1_.place(x=150, y=50)

        # Entry for user input
        self.Entry_1_ = Entry(main_frame, width=28, fg='black', bg='white', borderwidth=2, font=('Arial', 14, 'bold'))
        self.Entry_1_.insert(0, "Hello")  # Inserting default text
        self.Entry_1_.place(x=150, y=100)

        # Create a button for manual entry
        self.manual_button = Button(main_frame, text="Manual Entry", height=2, width=20, bg="blue", fg="white", command=self.manual_entry)
        self.manual_button.place(x=150, y=150)

        # Create a text widget for displaying conversation
        self.text = Text(main_frame, width=50, height=10, wrap=WORD)
        self.text.place(x=50, y=200)
        self.text.config(state=DISABLED)  # Make the text widget read-only

        # Add scroll bar
        scrollbar = Scrollbar(main_frame, orient=VERTICAL, command=self.text.yview)
        scrollbar.place(x=550, y=200, height=160)
        self.text.config(yscrollcommand=scrollbar.set)

        # Initialize conversation
        self.conversation_data = []

        # Define the intents
        self.intents = {
            "intents": [
                {
                    "tag": "google",
                    "patterns": [
                        "google",
                        "search",
                        "internet"
                    ],
                    "responses": [
                        "Redirecting to Google..."
                    ]
                },
                {
                    "tag": "greeting",
                    "patterns": [
                        "hi there",
                        "how are you",
                        "is anyone there?",
                        "hey",
                        "hola",
                        "hello",
                        "good day",
                        "namaste",
                        "yo"
                    ],
                    "responses": [
                        "Hi there, how can I help",
                        "Good to see you again",
                        "Hi there, how can I help?"
                    ]
                },
                {
                    "tag": "goodbye",
                    "patterns": [
                        "bye",
                        "see you later",
                        "goodbye",
                        "get lost",
                        "till next time",
                        "bbye"
                    ],
                    "responses": [
                        "See you!",
                        "Have a nice day",
                        "Bye! Come back again soon."
                    ]
                },
                {
                    "tag": "thanks",
                    "patterns": [
                        "thanks",
                        "thank you",
                        "that's helpful",
                        "awesome, thanks",
                        "thanks for helping me"
                    ],
                    "responses": [
                        "Happy to help!",
                        "Any time!",
                        "My pleasure"
                    ]
                },
                {
                    "tag": "noanswer",
                    "patterns": [],
                    "responses": [
                        "Sorry, can't understand you",
                        "Please give me more info",
                        "Not sure I understand"
                    ]
                },
                {
                    "tag": "options",
                    "patterns": [
                        "how you could help me?",
                        "what you can do?",
                        "what help you provide?",
                        "how you can be helpful?",
                        "what support is offered"
                    ],
                    "responses": [
                        "I am a general purpose chatbot. My capabilities are:\n"
                        "1. I can chat with you. Try asking me for jokes or riddles!\n"
                        "2. Ask me the date and time.\n"
                        "3. I can google search for you. Use format 'google: your query'.\n"
                        "4. I can get the present weather for any city. Use format 'weather: city name'.\n"
                        "5. I can get you the top 10 trending news in India. Use keywords 'Latest News'.\n"
                        "6. I can get you the top 10 trending songs globally. Type 'songs'.\n"
                        "7. I can set a timer for you. Enter 'set a timer: minutes to timer'.\n"
                        "8. I can get the present Covid stats for any country. Use 'covid 19: world' or 'covid 19: country name'.\n"
                        "For suggestions to help me improve, send an email to ted.thedlbot.suggestions@gmail.com. Thank you!!"
                    ]
                },
                {
                    "tag": "jokes",
                    "patterns": [
                        "tell me a joke",
                        "joke",
                        "make me laugh"
                    ],
                    "responses": [
                        "A perfectionist walked into a bar...apparently, the bar wasn't set high enough",
                        "I ate a clock yesterday, it was very time-consuming",
                        "Never criticize someone until you've walked a mile in their shoes. That way, when you criticize them, they won't be able to hear you from that far away. Plus, you'll have their shoes.",
                        "The world tongue-twister champion just got arrested. I hear they're gonna give him a really tough sentence.",
                        "I own the world's worst thesaurus. Not only is it awful, it's awful.",
                        "What did the traffic light say to the car? \"Don't look now, I'm changing.\"",
                        "What do you call a snowman with a suntan? A puddle.",
                        "How does a penguin build a house? Igloos it together",
                        "I went to see the doctor about my short-term memory problems – the first thing he did was make me pay in advance",
                        "As I get older and I remember all the people I’ve lost along the way, I think to myself, maybe a career as a tour guide wasn’t for me.",
                        "o what if I don't know what 'Armageddon' means? It's not the end of the world."
                    ]
                },
                {
                    "tag": "identity",
                    "patterns": [
                        "who are you",
                        "what are you"
                    ],
                    "responses": [
                        "I am Ted, a Deep-Learning chatbot"
                    ]
                },
                {
                    "tag": "datetime",
                    "patterns": [
                        "what is the time",
                        "what is the date",
                        "date",
                        "time",
                        "tell me the date",
                        "day",
                        "what day is it today"
                    ],
                    "responses": [
                        "Date and Time"
                    ]
                },
                {
                    "tag": "whatsup",
                    "patterns": [
                        "whats up",
                        "wazzup",
                        "how are you",
                        "sup",
                        "how you doing"
                    ],
                    "responses": [
                        "All good..What about you?"
                    ]
                },
                {
                    "tag": "haha",
                    "patterns": [
                        "haha",
                        "lol",
                        "rofl",
                        "lmao",
                        "thats funny"
                    ],
                    "responses": [
                        "Glad I could make you laugh!"
                    ]
                },
                {
                    "tag": "programmer",
                    "patterns": [
                        "who made you",
                        "who designed you",
                        "who programmed you"
                    ],
                    "responses": [
                        "I was made by Karan Malik."
                    ]
                },
                {
                    "tag": "insult",
                    "patterns": [
                        "you are dumb",
                        "shut up",
                        "idiot"
                    ],
                    "responses": [
                        "Well that hurts :("
                    ]
                },
                {
                    "tag": "activity",
                    "patterns": [
                        "what are you doing",
                        "what are you upto"
                    ],
                    "responses": [
                        "Talking to you, of course!"
                    ]
                },
                {
                    "tag": "exclaim",
                    "patterns": [
                        "awesome",
                        "great",
                        "i know",
                        "ok",
                        "yeah"
                    ],
                    "responses": [
                        "Yeah!"
                    ]
                },
                {
                    "tag": "weather",
                    "patterns": [
                        "temperature",
                        "weather",
                        "how hot is it"
                    ],
                    "responses": [
                        "..."
                    ]
                },
                {
                    "tag": "karan",
                    "patterns": [
                        "who is he",
                        "who is that",
                        "who is karan",
                        "karan malik"
                    ],
                    "responses": [
                        "Head over to his any of his social profiles to find out! Linkedin: www.linkedin.com/in/karan-malik-3a39191a7 Github: https://github.com/Karan-Malik"
                    ]
                },
                {
                    "tag": "contact",
                    "patterns": [
                        "contact developer",
                        "contact karan",
                        "contact programmer",
                        "contact creator"
                    ],
                    "responses": [            
                        "You can contact my creator at his Linkedin profile: www.linkedin.com/in/karan-malik-3a39191a7"
                    ]
                },
                {
                    "tag": "appreciate",
                    "patterns": [
                        "you are awesome",
                        "you are the best",
                        "you are great",
                        "you are good"
                    ],
                    "responses": [
                        "Thank you!"
                    ]
                },
                {
                    "tag": "nicetty",
                    "patterns": [
                        "it was nice talking to you",
                        "good talk"
                    ],
                    "responses": [
                        "It was nice talking to you as well! Come back soon!"
                    ]
                },
                {
                    "tag": "no",
                    "patterns": [
                        "no",
                        "nope"
                    ],
                    "responses": [
                        "ok"
                    ]
                },
                {
                    "tag": "news",
                    "patterns": [
                        "news",
                        "latest news",
                        "india news"
                    ],
                    "responses": [
                        "..."
                    ]
                },
                {
                    "tag": "inspire",
                    "patterns": [
                        "who inspires you",
                        "who is your inspiration",
                        "who motivates you"
                    ],
                    "responses": [
                        "Personally, I find Karan very inspiring. I might not be very fair though.."
                    ]
                },
                {
                    "tag": "cricket",
                    "patterns": [
                        "current cricket matches",
                        "cricket score"
                    ],
                    "responses": [
                        "..."
                    ]
                },
                {
                    "tag": "song",
                    "patterns": [
                        "top songs",
                        "best songs",
                        "hot songs",
                        "top 10 songs",
                        "top ten songs"
                    ],
                    "responses": [
                        "..."
                    ]
                },
                {
                    "tag": "greetreply",
                    "patterns": [
                        "i am good",
                        "i'm good",
                        "i am fine",
                        "i'm fine",
                        "good"
                    ],
                    "responses": [
                        "Good to know!"
                    ]
                },
                {
                    "tag": "timer",
                    "patterns": [
                        "set a timer"
                    ],
                    "responses": [
                        "..."
                    ]
                },
                {
                    "tag": "covid19",
                    "patterns": [
                        "covid 19"
                    ],
                    "responses": [
                        "..."
                    ]
                },
                {
                    "tag": "suggest",
                    "patterns": [
                        "you are useless",
                        "useless",
                        "suggest",
                        "suggestions",
                        "you are bad"
                    ],
                    "responses": [
                        "Please mail your suggestions to ted.thedlbot.suggestions@gmail.com. Thank you for helping me improve!"
                    ]
                },
                {
                    "tag": "riddle",
                    "patterns": [
                        "ask me a riddle",
                        "ask me a question",
                        "riddle"
                    ],
                    "responses": [
                        "What two things can you never eat for breakfast?.....Lunch and Dinner!",
                        "What word is spelled incorrectly in every single dictionary?.....Incorrectly",
                        "How can a girl go 25 days without sleep?.....She sleeps and night!",
                        "How do you make the number one disappear?.....Add the letter G and it’s 'gone'!",
                        "What will you actually find at the end of every rainbow?.....The letter 'w'",
                        "What can be caught but never thrown?.....A cold!",
                        "What has a thumb and four fingers but is not actually alive?.....Your Gloves!",
                        "What 5-letter word becomes shorter when you add two letters to it?.....Short",
                        "Why can't a bike stand on it's own?.....It is two-tired."
                    ]
                },
                {
                    "tag": "age",
                    "patterns": [
                        "how old are you",
                        "when were you made",
                        "what is your age"
                    ],
                    "responses": [
                        "I was made in 2024, if that's what you are asking!"
                    ]
                }
            ]
        }

    def manual_entry(self):
            # Get text from Entry_1_
            user_input = self.Entry_1_.get()
            # Initialize bot_response
            bot_response = "Sorry, I don't understand."

            # Check for matching intent
            found = False
            for intent in self.intents["intents"]:
                if user_input.lower() in intent["patterns"]:
                    bot_response = intent["responses"][0]  # Get the first response
                    found = True
                    break

            if not found:
                bot_response = "Sorry, I didn't get that. Could you try again?"

            # Append the conversation to the data
            self.conversation_data.append({"user": user_input, "bot": bot_response})

            # Display manual entry in the text widget
            self.text.config(state=NORMAL)
            for entry in self.conversation_data:
                self.text.insert(END, f"You: {entry['user']}\n")
                self.text.insert(END, f"Bot: {entry['bot']}\n\n")
            self.text.config(state=DISABLED)

if __name__ == "__main__":
    root = Tk()
    mb = Chatbot(root)
    root.mainloop()

    