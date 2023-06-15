import tkinter as tk

def login():
    # Retrieve the values from the username_entry and password_entry widgets
    username = username_entry.get()
    password = password_entry.get()

    # Perform authentication logic here
    # Validate username and password against database or predefined credentials

    if username == "admin" and password == "password":
        # Clear the login entries
        username_entry.delete(0, tk.END)
        password_entry.delete(0, tk.END)

        # Hide the login page and show the main menu
        login_frame.pack_forget()
        import main_menu
    else:
        # Show an error message for invalid login
        error_label.config(text="Invalid username or password")
        # Clear the login entries
        username_entry.delete(0, tk.END)
        password_entry.delete(0, tk.END)

# Create the main window
window = tk.Tk()
window.title("Criminal Recognition System")

# Login Page
login_frame = tk.Frame(window)

username_label = tk.Label(login_frame, text="Username:")
username_label.pack()

username_entry = tk.Entry(login_frame)
username_entry.pack()

password_label = tk.Label(login_frame, text="Password:")
password_label.pack()

password_entry = tk.Entry(login_frame, show="*")
password_entry.pack()

login_button = tk.Button(login_frame, text="Login", command=login)
login_button.pack()

error_label = tk.Label(login_frame, fg="red")
error_label.pack()

login_frame.pack()

# Start the Tkinter event loop
window.mainloop()
