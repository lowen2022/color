import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
import os
from pathlib import Path
import pyautogui


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master, bg="#BBE3F4")
        self.master = master
        self.master.title("Color Mixing App")
        self.master.geometry("550x550")
        self.pack(fill="both", expand=True)

        self.create_widgets()

    def create_widgets(self):
        # Add header label to the top of the GUI
        header_label = tk.Label(self, text="Hydro Kalon", font=("Courier new", 24, "bold"), bg="#BBE3F4", fg="white")
        header_label.pack(side="top", pady=20)

        # Create buttons for different modes
        self.mode1_btn = tk.Button(self, text="Mode 1: Input Color Values", font=("Courier new", 12), command=self.mode1, width=50, height=2, bg="#0099ff", fg="white", activebackground="#0066cc", activeforeground="white")
        self.mode1_btn.pack(side="top", pady=10)

        self.mode2_btn = tk.Button(self, text="Mode 2: Generate Mixed Color Possibilities",font=("Courier new", 12), command=self.mode2, width=50, height=2, bg="#0099ff", fg="white", activebackground="#0066cc", activeforeground="white")
        self.mode2_btn.pack(side="top", pady=10)

        self.mode3_btn = tk.Button(self, text="Mode 3: Color Extraction from an Image", font=("Courier new", 12), command=self.mode3, width=50, height=2, bg="#0099ff", fg="white", activebackground="#0066cc", activeforeground="white")
        self.mode3_btn.pack(side="top", pady=10)

    def go_home(self):
        # Destroy all widgets currently displayed
        for widget in self.winfo_children():
            widget.destroy()

        # Recreate the home screen widgets
        self.create_widgets()

    def mode1(self):
        # Destroy all widgets currently displayed
        for widget in self.winfo_children():
            widget.destroy()

        header_label = tk.Label(self, text="Mode 1", font=("Courier new", 24))
        header_label.pack(side="top", pady=20)

        self.home_btn = tk.Button(self)
        self.home_btn["text"] = "Home"
        self.home_btn["command"] = self.go_home
        self.home_btn.pack(side="top")

        # Labels and entries for color 1
        # Create a frame for Color 1
        color1_frame = tk.Frame(self)
        color1_frame.pack(padx=10, pady=10)

        # Labels and entries for color 1
        self.l1_label = tk.Label(color1_frame, text="L* for Color 1:")
        self.l1_label.pack(side="left")

        self.l1_entry = tk.Entry(color1_frame)
        self.l1_entry.pack(side="left", padx=5)

        self.a1_label = tk.Label(color1_frame, text="a* for Color 1:")
        self.a1_label.pack(side="left")

        self.a1_entry = tk.Entry(color1_frame)
        self.a1_entry.pack(side="left", padx=5)

        self.b1_label = tk.Label(color1_frame, text="b* for Color 1:")
        self.b1_label.pack(side="left")

        self.b1_entry = tk.Entry(color1_frame)
        self.b1_entry.pack(side="left", padx=5)

        self.q1_label = tk.Label(color1_frame, text="q* for Color 1:")
        self.q1_label.pack(side="left")

        self.q1_entry = tk.Entry(color1_frame)
        self.q1_entry.pack(side="left", padx=5)

        # Create a frame for Color 2
        color2_frame = tk.Frame(self)
        color2_frame.pack(padx=10, pady=10)

        # Labels and entries for color 2
        self.l2_label = tk.Label(color2_frame, text="L* for Color 2:")
        self.l2_label.pack(side="left")

        self.l2_entry = tk.Entry(color2_frame)
        self.l2_entry.pack(side="left", padx=5)

        self.a2_label = tk.Label(color2_frame, text="a* for Color 2:")
        self.a2_label.pack(side="left")

        self.a2_entry = tk.Entry(color2_frame)
        self.a2_entry.pack(side="left", padx=5)

        self.b2_label = tk.Label(color2_frame, text="b* for Color 2:")
        self.b2_label.pack(side="left")

        self.b2_entry = tk.Entry(color2_frame)
        self.b2_entry.pack(side="left", padx=5)

        self.q2_label = tk.Label(color2_frame, text="q* for Color 2:")
        self.q2_label.pack(side="left")

        self.q2_entry = tk.Entry(color2_frame)
        self.q2_entry.pack(side="left", padx=5)

        # Layout code to pack widgets
        self.l1_label.pack()
        self.l1_entry.pack()
        self.a1_label.pack()
        self.a1_entry.pack()
        self.b1_label.pack()
        self.b1_entry.pack()
        self.q1_label.pack()
        self.q1_entry.pack()

        self.l2_label.pack()
        self.l2_entry.pack()
        self.a2_label.pack()
        self.a2_entry.pack()
        self.b2_label.pack()
        self.b2_entry.pack()
        self.q2_label.pack()
        self.q2_entry.pack()

        # Button to trigger mixing  
        self.mix_button = tk.Button(self, text="Mix", command=self.mix_colors)
        self.mix_button.pack()

    def mix_colors(self):
        # Get input strings
        l1_str = self.l1_entry.get() 
        a1_str = self.a1_entry.get()
        b1_str = self.b1_entry.get()
        q1_str = self.q1_entry.get()

        l2_str = self.l2_entry.get()
        a2_str = self.a2_entry.get()
        b2_str = self.b2_entry.get()
        q2_str = self.q2_entry.get()

        # Validate inputs   
        if not all([l1_str, a1_str, b1_str, l2_str, a2_str, b2_str, q1_str, q2_str]):
            print("Please fill all fields")
            return

        try:
            # Convert to floats
            l1 = float(l1_str)
            a1 = float(a1_str)
            b1 = float(b1_str)
            q1 = float(q1_str)

            l2 = float(l2_str)
            a2 = float(a2_str)
            b2 = float(b2_str)
            q2 = float(q2_str)

        except ValueError:
            print("Invalid number input")
            return

        # Call mixing function
        mixed = self.mix(l1, a1, b1, q1, l2, a2, b2, q2)

        # Display mixed color
        mixed_label = tk.Label(self, text=f"Mixed Color: {mixed}", font=("Courier new", 16))
        mixed_label.pack()

    def mix(self, l1, a1, b1, q1, l2, a2, b2, q2):
        # Calculate the average of L*, a*, b*, and q values
        l_avg = (l1 + l2) / 2
        a_avg = (a1 + a2) / 2
        b_avg = (b1 + b2) / 2
        q_avg = (q1 + q2) / 2

        # Return mixed color as a tuple
        return (l_avg, a_avg, b_avg, q_avg)


    def mode2(self):
        # Destroy all widgets currently displayed
        for widget in self.winfo_children():
            widget.destroy()

        header_label = tk.Label(self, text="Mode 2", font=("Courier new", 24))
        header_label.pack(side="top", pady=20)

        self.home_btn = tk.Button(self)
        self.home_btn["text"] = "Home"
        self.home_btn["command"] = self.go_home
        self.home_btn.pack(side="top")

        # Create new widgets for Mode 2
        self.color_label = tk.Label(self, text="Enter color values:")
        self.color_label.pack(side="top")

        self.l_label = tk.Label(self, text="L*:")
        self.l_label.pack(side="top")
        self.l_entry= tk.Entry(self)
        self.l_entry.pack(side="top")

        self.a_label = tk.Label(self, text="a*:")
        self.a_label.pack(side="top")
        self.a_entry = tk.Entry(self)
        self.a_entry.pack(side="top")

        self.b_label = tk.Label(self, text="b*:")
        self.b_label.pack(side="top")
        self.b_entry = tk.Entry(self)
        self.b_entry.pack(side="top")

        self.num_mixes_label = tk.Label(self, text="Number of mixed colors to generate:")
        self.num_mixes_label.pack(side="top")

        self.num_mixes_entry = tk.Entry(self)
        self.num_mixes_entry.pack(side="top")

        self.calculate_btn = tk.Button(self)
        self.calculate_btn["text"] = "Generate"
        self.calculate_btn["command"] = self.generate_mixed_colors
        self.calculate_btn.pack(side="top")

    def generate_mixed_colors(self):
        l = float(self.l_entry.get())
        a = float(self.a_entry.get())
        b = float(self.b_entry.get())
        num_mixes = int(self.num_mixes_entry.get())

        # Generate mixed color possibilities
        mixed_colors = self.generate_mixes(l, a, b, num_mixes)

        # Display mixed color possibilities in the text area
        self.text_area.delete('1.0', tk.END)  # Clear the text area
        for i, color in enumerate(mixed_colors):
            self.text_area.insert(tk.END, f"Option {i+1}: {color}\n")

    def generate_mixes(self, l, a, b, num_mixes):
        # Generate mixed color possibilities and return them as a list
        mixed_colors = []
        for i in range(num_mixes):
            mixed_l = l * (i+1) / num_mixes
            mixed_a = a * (i+1) / num_mixes
            mixed_b = b * (i+1) / num_mixes
            mixed_color = (mixed_l, mixed_a, mixed_b)
            mixed_colors.append(mixed_color)
        return mixed_colors

    def mode3(self):
        # Destroy all widgets currently displayed
        for widget in self.winfo_children():
            widget.destroy()
        
        header_label = tk.Label(self, text="Mode 3", font=("Courier new", 24))
        header_label.pack(side="top", pady=20)

        self.home_btn = tk.Button(self)
        self.home_btn["text"] = "Home"
        self.home_btn["command"] = self.go_home
        self.home_btn.pack(side="top")

        # Create new widgets for Mode 3
        self.image_label = tk.Label(self, text="Select an image:")
        self.image_label.pack(side="top", padx=10)

        self.canvas = tk.Canvas(self, width=400, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.color_label = tk.Label(self, text="Selected Color: ")
        self.color_label.pack()

        self.import_button = tk.Button(self, text="Import Image", command=self.import_image)
        self.import_button.pack()

        self.color_label = tk.Label(self, text="Extracted color values:")
        self.color_label.pack(side="top")

        self.l_value_label = tk.Label(self, bg="#BBE3F4")
        self.l_value_label.pack(side="top")

        self.a_value_label = tk.Label(self, bg="#BBE3F4")
        self.a_value_label.pack(side="top")

        self.b_value_label = tk.Label(self, bg="#BBE3F4")
        self.b_value_label.pack(side="top")

    def choose_color(self, event):

        x, y = event.x, event.y
        pixel_rgb = self.image.getpixel((x, y))
        self.color_label.config(text=f"Selected Color: {pixel_rgb}")
        color_format = "#{:02x}{:02x}{:02x}".format(pixel_rgb[0], pixel_rgb[1], pixel_rgb[2])

        block_color = color_format
        block_width = 50
        block_height = 50
        block_x = 200  # Adjust the x-coordinate
        block_y = 550 # Adjust the y-coordinate

        self.canvas.create_rectangle(block_x, block_y, block_x + block_width, block_y + block_height, fill=block_color)
    def import_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png; *.bmp")])
        if file_path:
            self.image = Image.open(file_path)
            original_width, original_height = self.image.size

            if original_width < 200 and original_height < 200:
                self.image = self.image.resize((original_width * 2, original_height * 2))
            elif original_width > 900 or original_height > 900:
                scaling_factor = min(900 / original_width, 900 / original_height)
                new_width = int(original_width * scaling_factor)
                new_height = int(original_height * scaling_factor)
                self.image.thumbnail((new_width, new_height))

            photo = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=500, height=500)
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo
            self.canvas.bind("<Motion>", self.choose_color)

root = tk.Tk()
app = Application(master=root)
try:
    app.mainloop()
except KeyboardInterrupt:
    pass
