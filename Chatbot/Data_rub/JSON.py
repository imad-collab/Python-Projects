import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Text, Entry, Frame, Button, Label, DISABLED, END
import json
import random

class Chatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot Application")
        self.root.geometry("600x400+300+100")

        # Create a Main Frame (White Color)
        main_frame = Frame(self.root, bg="white", width=600, height=400)
        main_frame.pack(fill=tk.BOTH, expand=True)

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

        # Create a button for uploading JSON file
        self.upload_button = Button(main_frame, text="Upload Intents JSON", height=2, width=20, bg="green", fg="white", command=self.upload_json_file)
        self.upload_button.place(x=150, y=200)

        # Create a text widget for displaying conversation
        self.text = Text(main_frame, width=50, height=10, wrap=tk.WORD)
        self.text.place(x=50, y=250)
        self.text.config(state=DISABLED)  # Make the text widget read-only

        # Add scroll bar
        scrollbar = Scrollbar(main_frame, orient=tk.VERTICAL, command=self.text.yview)
        scrollbar.place(x=550, y=250, height=160)
        self.text.config(yscrollcommand=scrollbar.set)

        # Initialize conversation
        self.conversation_data = []

        # Initialize intents as an empty list
        self.intents = []

    def manual_entry(self):
        # Get text from Entry_1_
        user_input = self.Entry_1_.get().lower()  # Convert user input to lowercase
        # Initialize bot_response
        bot_response = "Sorry, I don't understand."

        # Check for matching intent
        response_found = False  # Initialize response_found as False
        for intent in self.intents:
            for pattern in intent["patterns"]:
                if pattern in user_input:  # Check if pattern is in user input
                    bot_response = random.choice(intent["responses"])  # Choose a random response
                    response_found = True
                    break
            if response_found:
                break

        if not response_found:
            bot_response = "Sorry, I didn't get that. Could you try again?"

        # Append the conversation to the data
        self.conversation_data.append({"user": user_input, "bot": bot_response})

        # Display manual entry in the text widget
        self.text.config(state=tk.NORMAL)
        self.text.delete('1.0', tk.END)  # Clear the text widget
        for entry in self.conversation_data:
            self.text.insert(tk.END, f"You: {entry['user']}\n")
            self.text.insert(tk.END, f"Bot: {entry['bot']}\n\n")
        self.text.config(state=tk.DISABLED)

    def upload_json_file(self):
        # Open a file dialog to select a JSON file
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    self.intents = json.load(file)["intents"]
                    messagebox.showinfo("Success", "Intents loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load intents: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    mb = Chatbot(root)
    root.mainloop()
