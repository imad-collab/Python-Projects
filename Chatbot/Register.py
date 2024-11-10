from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import mysql.connector


class Register:
    def __init__(self, root):
        self.root = root
        self.root.title("Register")
        self.root.geometry("1600x900+140+65")

        # Create database and tables if not exists
        self.create_database_and_tables()

        # Frame for the form
        self.frame = Frame(self.root, bg="black")
        self.frame.place(x=200, y=150, width=800, height=600)

        # Create form fields
        self.create_form(self.frame)

    def create_database_and_tables(self):
        try:
            connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="hydra12r"
            )
            cursor = connection.cursor()

            # Create database if not exists
            cursor.execute("CREATE DATABASE IF NOT EXISTS chatbot_db")

            # Use the database
            cursor.execute("USE chatbot_db")

            # Create table Register if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Register (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    contact VARCHAR(20) NOT NULL,
                    email VARCHAR(255) NOT NULL,
                    address TEXT NOT NULL
                )
            """)

            # Create table users if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) NOT NULL,
                    password VARCHAR(255) NOT NULL
                )
            """)

            connection.commit()
            print("Database and tables created successfully")

        except mysql.connector.Error as err:
            messagebox.showerror("Error", f"Error creating database and tables: {err}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def create_form(self, frame):
        username_label = Label(frame, text="Username", font=("Arial", 14, "bold"), fg="gray", bg="black")
        username_label.place(x=50, y=50)
        self.username_entry = Entry(frame, width=30, fg='black', bg='white', font=("Arial", 14))
        self.username_entry.place(x=200, y=50)

        password_label = Label(frame, text="Password", font=("Arial", 14, "bold"), fg="gray", bg="black")
        password_label.place(x=50, y=100)
        self.password_entry = Entry(frame, width=30, fg='black', bg='white', font=("Arial", 14), show='*')
        self.password_entry.place(x=200, y=100)

        contact_label = Label(frame, text="Contact No", font=("Arial", 14, "bold"), fg="gray", bg="black")
        contact_label.place(x=50, y=150)
        self.contact_entry = Entry(frame, width=30, fg='black', bg='white', font=("Arial", 14))
        self.contact_entry.place(x=200, y=150)

        email_label = Label(frame, text="Email ID", font=("Arial", 14, "bold"), fg="gray", bg="black")
        email_label.place(x=50, y=200)
        self.email_entry = Entry(frame, width=30, fg='black', bg='white', font=("Arial", 14))
        self.email_entry.place(x=200, y=200)

        address_label = Label(frame, text="Address", font=("Arial", 14, "bold"), fg="gray", bg="black")
        address_label.place(x=50, y=250)
        self.address_entry = Entry(frame, width=30, fg='black', bg='white', font=("Arial", 14))
        self.address_entry.place(x=200, y=250)

        register_button = Button(frame, text="Register", font=("Arial", 14, "bold"), fg="black", bg="gray", command=self.register)
        register_button.place(x=200, y=300)

    def register(self):
        
        username = self.username_entry.get()
        password = self.password_entry.get()
        contact = self.contact_entry.get()
        email = self.email_entry.get()
        address = self.address_entry.get()

        if not all([username, password, contact, email, address]):
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

            # Insert into Register table
            cursor.execute(
                "INSERT INTO Register (username, password, contact, email, address) VALUES (%s, %s, %s, %s, %s)",
                (username, password, contact, email, address)
            )

            # Insert into users table
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, password)
            )

            connection.commit()
            connection.close()

            messagebox.showinfo("Success", "Registration successful")
        except mysql.connector.Error as err:
            messagebox.showerror("Error", f"Error inserting data into database: {err}")

if __name__ == "__main__":
    root = Tk()
    app = Register(root)
    root.mainloop()


