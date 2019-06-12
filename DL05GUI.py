##########################################################
# I WILL BE IMPORTING THIS GUI CODE WITHIN THE CHATBOT CODE
# WORKING ON GETTING THE CHATBOT WORKING PROPERLY BEFORE I WORRY ABOUT THE GUI
##########################################################
# Libraries
import tkinter as tk
from tkinter import *
# Window Tabs Libraries
from tkinter import ttk
from tkinter.scrolledtext import *

# Create GUI
# Create Window
# Build main window
window = tk.Tk()
# Main Title
window.title("Sample Window Title")
# Window Size
window.geometry("680x800")

# Window Tabs
# Set style of tabs
style = ttk.Style(window)
# Set location of tabs
# wn = West North
# ws = West South
style.configure('lefttab.TNotebook', tabpostition='wn')

tab_control = ttk.Notebook(window)

# Create tabs
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)
tab5 = ttk.Frame(tab_control)

# Add tabs to window
tab_control.add(tab1, text='Home tab')
tab_control.add(tab2, text='Another tab')
tab_control.add(tab3, text='Another tab')
tab_control.add(tab4, text='Another tab')
tab_control.add(tab5, text='About tab')

# Create Labels
# Place Labels
label1 = Label(tab1, text='Home', padx=5, pady=5)
# 0,0 is top left of window
label1.grid(column=0, row=0)
label2 = Label(tab2, text='My Page Title', padx=5, pady=5)
label2.grid(column=0, row=0)
label3 = Label(tab3, text='My Page Title', padx=5, pady=5)
label3.grid(column=0, row=0)
label4 = Label(tab4, text='My Page Title', padx=5, pady=5)
label4.grid(column=0, row=0)
label5 = Label(tab5, text='About Title', padx=5, pady=5)
label5.grid(column=0, row=0)

tab_control.pack(expand=1, fill='both')

# Functions
def run_code_on_tab_1():
    input_text = input_box.get('1.0', tk.END)
    output_text = input_text
    processed_text = input_text
    tab1_display.insert(tk.END, processed_text)


def clear_input_box():
    input_box.delete(1.0, END)


def clear_display_result():
    tab1_display.delete(1.0,END)


# Main Home Tab
l1 = Label(tab1, text='Enter Text to Process in some way...', padx=5, pady=5)
l1.grid(row=1, column=0)
input_box = ScrolledText(tab1, height=10)
input_box.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# Button
button_reset_input = Button(tab1, text='Reset Input', command=clear_input_box, width=12, bg='purple', fg='#fff')
button_reset_input.grid(row=3, column=0, padx=10, pady=10)

button_process_stuff = Button(tab1, text="Run me", command=run_code_on_tab_1, width=12, bg='#25d366', fg='purple')
button_process_stuff.grid(row=4, column=0, padx=10, pady=10)

button_clear_output = Button(tab1, text='Clear Output', command=clear_display_result, width=12, bg='purple', fg='#fff')
button_clear_output.grid(row=5, column=0, padx=10, pady=10)

# Display Screen for Result
tab1_display = ScrolledText(tab1)
tab1_display.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

# Keep window alive
window.mainloop()