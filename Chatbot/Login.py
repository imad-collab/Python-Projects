from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import mysql.connector
from Register import Register 
from chatbot import Chatbot

class Login_Window:
    def __init__(self, root):
        self.root = root
        self.root.title("Login_Page")
        self.root.geometry("700x500+550+250")

        #====================================FrameWork=================================
        
        # Background image
        self.bg = ImageTk.PhotoImage(file=r"D:\Chatbot\Images\white_image.jpg")
        lbl_bg = Label(self.root, image=self.bg)
        lbl_bg.place(x=0, y=0, relwidth=1, relheight=1)

        # Frame for the login section
        frame = Frame(self.root, bg="black")
        frame.place(x=0, y=0, width=700, height=500)

        # Create a Label
        get_str = Label(frame, text="Get Started", font=("Arial", 20, "bold"), fg="white", bg="black")
        get_str.place(x=95, y=100)

        # Create Label 1 User Name 
        user_name = Label(frame, text="Username", font=("Arial", 20, "bold"), fg="white", bg="black")
        user_name.place(x=70, y=155)

        # Create Label 2 Password 
        password = Label(frame, text="Password", font=("Arial", 20, "bold"), fg="white", bg="black")
        password.place(x=70, y=210)

        # Create Entry Field 1
        self.entry_Field_1 = Entry(frame, width=25, fg='black', bg='white', font=("Arial", 20, "bold"))
        self.entry_Field_1.place(x=250, y=155)

        # Create Entry Field 2
        self.entry_Field_2 = Entry(frame, width=25, fg='black', bg='white', font=("Arial", 20, "bold"), show='*')
        self.entry_Field_2.place(x=250, y=210)

        # Create a Login Button 1
        Login_button_1 = Button(frame, text="Login", fg="white", bg='black', width=15, height=2, font=("Georgia", 12, "bold"), command=self.login)
        Login_button_1.place(x=70, y=300)

        # Create a Register Button 2
        register_button_2 = Button(frame, text="Register", fg="white", bg='black', width=15, height=2, font=("Georgia", 12, "bold"),command=self.open_register_window)
        register_button_2.place(x=250, y=300)

        # Create a Forgetpassword Button 3
        Forgetpassword_button_3 = Button(frame, text="Forget Password?", fg="white", bg='black', width=15, height=2, font=("Georgia", 12, "bold"))
        Forgetpassword_button_3.place(x=430, y=300)


        #==================================== FrameWork End =================================


    def login(self):
        username = self.entry_Field_1.get()
        password = self.entry_Field_2.get()
        
        if username == "" or password == "":
            messagebox.showerror("Error", "All fields are required")
            return

        try:
            connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="hydra12r",
                database="chatbot_db"
            )

            cursor = connection.cursor()
            cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
            row = cursor.fetchone()

            if row:
                messagebox.showinfo("Success", "Credentials are verified.")
                self.open_chatbot_window()  # Open the chatbot window upon successful login
            else:
                messagebox.showerror("Invalid", "Invalid username & password")

            connection.close()
        except mysql.connector.Error as err:
            messagebox.showerror("Error", f"Error connecting to the database: {err}")

    def open_register_window(self):
        self.new_window = Toplevel(self.root)
        self.app = Register(self.new_window)

    def open_chatbot_window(self):
        self.new_window = Toplevel(self.root)
        self.app = Chatbot(self.new_window) 

if __name__ == "__main__":
    root = Tk()
    app = Login_Window(root)
    root.mainloop()
